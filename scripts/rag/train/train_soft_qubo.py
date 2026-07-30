"""Train Soft QUBO with Recall loss (Idea 7).

End-to-end optimization of QUBO objective using task loss instead of
heuristic quality/redundancy signals.

Usage:
    # Train on 10 samples (quick validation)
    python -m scripts.idea7.train_soft_qubo \\
        --max_samples 10 \\
        --epochs 50 \\
        --output_dir exchange/idea7_mvp

    # Full training (200 samples, 2 hours)
    python -m scripts.idea7.train_soft_qubo \\
        --max_samples 200 \\
        --epochs 100 \\
        --output_dir exchange/idea7_full
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from applications.rag.data import load_dataset_for_rag, make_corpus_manager
from applications.rag.retrieval import make_encoder
from applications.rag.selector import select_passages
from qore.soft_qubo import SoftQUBO, LearnableQUBO, compute_recall_loss, soft_select_to_hard


def parse_args():
    p = argparse.ArgumentParser(description="Train Soft QUBO with Recall loss")

    # Dataset
    p.add_argument("--dataset", default="nq_open", help="Dataset name")
    p.add_argument("--split", default="validation", help="Dataset split")
    p.add_argument("--max_samples", type=int, default=10, help="Training samples")

    # Corpus
    p.add_argument("--corpus_mode", default="aligned", help="Corpus mode")
    p.add_argument("--corpus_output_dir", help="Cache dir for aligned corpus")
    p.add_argument("--n_distractors", type=int, default=36000, help="Distractors")

    # Retrieval
    p.add_argument("--encoder_type", default="sentence", help="Encoder type")
    p.add_argument("--top_k_retrieval", type=int, default=50, help="Retrieve top-K")

    # Selection
    p.add_argument("--K", type=int, default=5, help="Select K passages")

    # Training
    p.add_argument("--model_type", default="learnable",
                   choices=["soft", "learnable"],
                   help="Model type: 'soft' (fixed weights) or 'learnable' (learn weights)")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--temperature_init", type=float, default=1.0, help="Initial temperature")
    p.add_argument("--temperature_final", type=float, default=0.3, help="Final temperature")
    p.add_argument("--temperature_anneal_epochs", type=int, default=30,
                   help="Anneal temperature over this many epochs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    p.add_argument("--output_dir", required=True, help="Output directory")
    p.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")

    return p.parse_args()


def answer_has_match_in_text(answer: str, text: str) -> bool:
    """Check if answer appears in text with token boundaries."""
    import re
    answer_norm = answer.lower().strip()
    text_norm = text.lower()
    if answer_norm not in text_norm:
        return False
    pattern = r'\b' + re.escape(answer_norm) + r'\b'
    return re.search(pattern, text_norm) is not None


def prepare_sample(
    sample: dict,
    corpus_manager,
    encoder,
    top_k_retrieval: int,
) -> dict | None:
    """Prepare one sample for training.

    Returns:
        dict with keys: question, query_emb, passage_embs, passage_texts,
                       gold_indices, retrieved_indices
        None if retrieval fails (gold not in top-K)
    """
    question = sample["question"]
    gold_answers = sample["answers"]
    question_id = sample.get("id", sample.get("question_id"))

    # Encode query
    query_emb = encoder.encode_queries([question])[0]

    # Retrieve candidates
    retrieved_indices, scores = corpus_manager.retrieve(query_emb, top_k=top_k_retrieval)
    if retrieved_indices is None or len(retrieved_indices) == 0:
        return None

    # Get passage texts and embeddings
    corpus = corpus_manager.corpus
    passage_texts = [corpus.passages[i] for i in retrieved_indices]
    passage_embs = corpus.embeddings[retrieved_indices]

    # Find gold indices in retrieved set
    gold_corpus_indices = corpus.gold_for(question_id)
    gold_indices = []
    for i, corpus_idx in enumerate(retrieved_indices):
        if corpus_idx in gold_corpus_indices:
            gold_indices.append(i)

    if len(gold_indices) == 0:
        # Retrieval failure: gold not in candidates
        return None

    return {
        "question": question,
        "query_emb": query_emb,
        "passage_embs": passage_embs,
        "passage_texts": passage_texts,
        "gold_indices": gold_indices,
        "retrieved_indices": retrieved_indices,
    }


def compute_hard_recall(p: torch.Tensor, gold_indices: list[int], K: int) -> float:
    """Compute hard Recall@K from soft selection."""
    x = soft_select_to_hard(p, K)
    selected = torch.where(x == 1)[0].cpu().numpy()
    gold_set = set(gold_indices)
    hits = len(gold_set & set(selected))
    return hits / len(gold_set)


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: list[dict],
    K: int,
    device: torch.device,
    temperature: float,
) -> dict:
    """Train for one epoch."""
    model.train()
    if hasattr(model, 'soft_qubo'):
        model.soft_qubo.temperature = temperature
    else:
        model.temperature = temperature

    epoch_loss = 0.0
    epoch_recall = 0.0
    n_samples = 0

    for sample in train_data:
        # Convert to torch
        query_emb = torch.from_numpy(sample["query_emb"]).float().to(device)
        passage_embs = torch.from_numpy(sample["passage_embs"]).float().to(device)
        gold_indices = torch.tensor(sample["gold_indices"], dtype=torch.long, device=device)

        N = len(passage_embs)

        # Compute quality and redundancy (classical signals)
        # Quality: cosine similarity with query
        query_norm = query_emb / (torch.norm(query_emb) + 1e-12)
        passage_norms = passage_embs / (torch.norm(passage_embs, dim=1, keepdim=True) + 1e-12)
        a = (passage_norms @ query_norm).clamp(min=0.0)  # quality scores

        # Redundancy: pairwise cosine similarity
        b = passage_norms @ passage_norms.T
        b = b.clamp(min=0.0, max=1.0)
        b.fill_diagonal_(0.0)

        # Normalize quality to [0, 1]
        if a.max() > a.min():
            a = (a - a.min()) / (a.max() - a.min())

        # Forward pass
        if isinstance(model, LearnableQUBO):
            p, info = model(a, b, K)
        else:
            p, info = model(a, b, K, lam=2.0, gamma=1.0)

        # Compute loss
        loss = compute_recall_loss(p, gold_indices)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        epoch_loss += loss.item()
        hard_recall = compute_hard_recall(p, sample["gold_indices"], K)
        epoch_recall += hard_recall
        n_samples += 1

    return {
        "loss": epoch_loss / n_samples,
        "recall": epoch_recall / n_samples,
        "n_samples": n_samples,
    }


def evaluate(
    model: torch.nn.Module,
    eval_data: list[dict],
    K: int,
    device: torch.device,
) -> dict:
    """Evaluate model on validation data."""
    model.eval()

    total_loss = 0.0
    total_recall = 0.0
    n_samples = 0

    with torch.no_grad():
        for sample in eval_data:
            query_emb = torch.from_numpy(sample["query_emb"]).float().to(device)
            passage_embs = torch.from_numpy(sample["passage_embs"]).float().to(device)
            gold_indices = torch.tensor(sample["gold_indices"], dtype=torch.long, device=device)

            # Compute signals
            query_norm = query_emb / (torch.norm(query_emb) + 1e-12)
            passage_norms = passage_embs / (torch.norm(passage_embs, dim=1, keepdim=True) + 1e-12)
            a = (passage_norms @ query_norm).clamp(min=0.0)
            b = passage_norms @ passage_norms.T
            b = b.clamp(min=0.0, max=1.0)
            b.fill_diagonal_(0.0)

            if a.max() > a.min():
                a = (a - a.min()) / (a.max() - a.min())

            # Forward
            if isinstance(model, LearnableQUBO):
                p, info = model(a, b, K)
            else:
                p, info = model(a, b, K, lam=2.0, gamma=1.0)

            # Metrics
            loss = compute_recall_loss(p, gold_indices)
            total_loss += loss.item()
            hard_recall = compute_hard_recall(p, sample["gold_indices"], K)
            total_recall += hard_recall
            n_samples += 1

    return {
        "loss": total_loss / n_samples,
        "recall": total_recall / n_samples,
        "n_samples": n_samples,
    }


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"💾 Output directory: {output_dir}")

    # Load dataset
    print(f"📚 Loading dataset: {args.dataset} ({args.split})")
    samples = load_dataset_for_rag(args.dataset, args.split)

    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"   Loaded {len(samples)} samples")

    # Setup encoder (needed for aligned mode)
    print(f"🔤 Loading encoder: {args.encoder_type}")
    encoder = make_encoder(args.encoder_type)

    # Setup corpus
    print(f"🗂️  Setting up corpus (mode={args.corpus_mode})")
    corpus_config = {
        "output_dir": args.corpus_output_dir,
        "n_distractors": args.n_distractors,
        "seed": args.seed,
    }
    if args.corpus_mode == "aligned":
        corpus_config["embedder"] = encoder.encode_passages

    corpus_manager = make_corpus_manager(args.corpus_mode, corpus_config)

    # Build corpus
    print(f"   Building corpus for {len(samples)} questions...")
    corpus = corpus_manager.build(samples)
    print(f"   Corpus ready: {len(corpus)} passages")

    # Prepare training data
    print(f"⚙️  Preparing training data...")
    train_data = []
    for sample in tqdm(samples, desc="Preparing"):
        prepared = prepare_sample(sample, corpus_manager, encoder, args.top_k_retrieval)
        if prepared is not None:
            train_data.append(prepared)

    print(f"   ✅ {len(train_data)}/{len(samples)} samples valid (gold in top-{args.top_k_retrieval})")

    if len(train_data) == 0:
        print("❌ No valid training samples! Check retrieval setup.")
        return

    # Split train/val (80/20)
    n_train = int(0.8 * len(train_data))
    train_samples = train_data[:n_train]
    val_samples = train_data[n_train:]

    print(f"   Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Create model
    if args.model_type == "learnable":
        model = LearnableQUBO(
            temperature=args.temperature_init,
            init_w_a=1.0,
            init_w_b=1.0,
            learn_lam=False,
        )
        print(f"🧠 Model: LearnableQUBO (learns quality/redundancy weights)")
    else:
        model = SoftQUBO(temperature=args.temperature_init)
        print(f"🧠 Model: SoftQUBO (fixed weights)")

    model = model.to(device)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs")
    print(f"   Temperature: {args.temperature_init} → {args.temperature_final} over {args.temperature_anneal_epochs} epochs")

    best_val_recall = 0.0
    history = []

    for epoch in range(args.epochs):
        # Anneal temperature
        if epoch < args.temperature_anneal_epochs:
            progress = epoch / args.temperature_anneal_epochs
            temperature = args.temperature_init * (1 - progress) + args.temperature_final * progress
        else:
            temperature = args.temperature_final

        # Train
        train_metrics = train_epoch(model, optimizer, train_samples, args.K, device, temperature)

        # Validate
        if len(val_samples) > 0:
            val_metrics = evaluate(model, val_samples, args.K, device)
        else:
            val_metrics = {"loss": 0.0, "recall": 0.0}

        # Log
        epoch_info = {
            "epoch": epoch + 1,
            "temperature": temperature,
            "train_loss": train_metrics["loss"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_metrics["loss"],
            "val_recall": val_metrics["recall"],
        }

        if isinstance(model, LearnableQUBO):
            epoch_info["w_a"] = np.exp(model.log_w_a.item())
            epoch_info["w_b"] = np.exp(model.log_w_b.item())

        history.append(epoch_info)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"T={temperature:.3f} | "
              f"Loss={train_metrics['loss']:.4f} | "
              f"Recall={train_metrics['recall']:.4f} | "
              f"Val_Recall={val_metrics['recall']:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }
            torch.save(checkpoint, output_dir / f"checkpoint_epoch{epoch+1}.pt")

        # Save best model
        if val_metrics["recall"] > best_val_recall:
            best_val_recall = val_metrics["recall"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "recall": best_val_recall,
            }, output_dir / "best_model.pt")

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"   Best validation Recall: {best_val_recall:.4f}")
    print(f"   Model saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
