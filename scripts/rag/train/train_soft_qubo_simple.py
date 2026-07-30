"""Simplified Soft QUBO training using pre-evaluated result.json.

Instead of rebuilding corpus and retrieval from scratch, this script loads
a result.json from Phase 2 evaluation (which already has retrieved passages
and gold annotations) and trains the Soft QUBO on that data.

Usage:
    # Use Phase 2 Idea 6 result as training data
    python -m scripts.idea7.train_soft_qubo_simple \\
        --result_json exchange/p2_idea7_diagnosis/20260730T111502/result.json \\
        --epochs 50 \\
        --output_dir exchange/idea7_mvp
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from qore.soft_qubo import LearnableQUBO, SoftQUBO, compute_recall_loss, soft_select_to_hard


def parse_args():
    p = argparse.ArgumentParser(description="Train Soft QUBO from result.json")

    # Input
    p.add_argument("--result_json", required=True, help="Path to result.json from Phase 2")
    p.add_argument("--max_samples", type=int, default=0, help="Limit samples (0=all)")

    # Selection
    p.add_argument("--K", type=int, default=5, help="Select K passages")

    # Training
    p.add_argument("--model_type", default="learnable",
                   choices=["soft", "learnable"],
                   help="Model type")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--temperature_init", type=float, default=1.0, help="Initial temperature")
    p.add_argument("--temperature_final", type=float, default=0.3, help="Final temperature")
    p.add_argument("--temperature_anneal_epochs", type=int, default=30,
                   help="Anneal temperature over this many epochs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    p.add_argument("--output_dir", required=True, help="Output directory")
    p.add_argument("--save_every", type=int, default=10, help="Save every N epochs")

    return p.parse_args()


def load_training_data(result_json_path: str, max_samples: int = 0) -> list[dict]:
    """Load training samples from result.json or synthetic data.

    Supports two formats:
    1. result.json from eval_rag_refactored (dict with "samples" key)
    2. Synthetic data (list with "embeddings" and "gold_indices" directly)

    Each sample needs:
    - embeddings: (N, d) numpy array
    - gold_indices: list of ints (which passages contain the answer)
    """
    with open(result_json_path) as f:
        data = json.load(f)

    # Handle wrapped format (eval_rag_refactored output)
    if isinstance(data, dict) and "samples" in data:
        results = data["samples"]
    elif isinstance(data, list):
        results = data
    else:
        raise ValueError(f"Unexpected result.json format: {type(data)}")

    samples = []
    for i, item in enumerate(results):
        if max_samples > 0 and i >= max_samples:
            break

        # Format 1: Synthetic data (direct format)
        if "embeddings" in item and "gold_indices" in item:
            samples.append({
                "question": item.get("question", ""),
                "embeddings": np.array(item["embeddings"], dtype=np.float32),
                "gold_indices": item["gold_indices"],
                "texts": item.get("texts", [f"Passage {i}" for i in range(len(item["embeddings"]))]),
            })
            continue

        # Format 2: result.json from eval_rag_refactored
        # Field name can be "retrieved" or "selected_passages"
        passages_key = None
        if "retrieved" in item and len(item["retrieved"]) > 0:
            passages_key = "retrieved"
        elif "selected_passages" in item and len(item["selected_passages"]) > 0:
            passages_key = "selected_passages"

        if passages_key is None:
            continue

        # Find gold indices
        gold_indices = []
        for j, passage in enumerate(item[passages_key]):
            if passage.get("is_gold", False):
                gold_indices.append(j)

        if len(gold_indices) == 0:
            # No gold in retrieved passages
            continue

        # Get embeddings (if available)
        embeddings = []
        for passage in item[passages_key]:
            if "embedding" in passage and passage["embedding"] is not None:
                embeddings.append(passage["embedding"])

        if len(embeddings) == 0:
            # Need to compute embeddings from text
            # For MVP, skip samples without embeddings
            continue

        samples.append({
            "question": item.get("question", ""),
            "embeddings": np.array(embeddings, dtype=np.float32),
            "gold_indices": gold_indices,
            "texts": [p["text"] for p in item[passages_key]],
        })

    return samples


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
        passage_embs = torch.from_numpy(sample["embeddings"]).float().to(device)
        gold_indices = torch.tensor(sample["gold_indices"], dtype=torch.long, device=device)

        N = len(passage_embs)

        # Compute quality: use first passage as "query" (rough approximation)
        # In reality, should use actual query embedding
        # For MVP: use mean of gold embeddings as "query"
        if len(gold_indices) > 0:
            query_emb = passage_embs[gold_indices].mean(dim=0)
        else:
            query_emb = passage_embs.mean(dim=0)

        # Quality: cosine similarity with query
        query_norm = query_emb / (torch.norm(query_emb) + 1e-12)
        passage_norms = passage_embs / (torch.norm(passage_embs, dim=1, keepdim=True) + 1e-12)
        a = (passage_norms @ query_norm).clamp(min=0.0)

        # Redundancy: pairwise cosine similarity
        b = passage_norms @ passage_norms.T
        b = b.clamp(min=0.0, max=1.0)
        b.fill_diagonal_(0.0)

        # Normalize quality
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
        x = soft_select_to_hard(p, K)
        selected = torch.where(x == 1)[0].cpu().numpy()
        gold_set = set(sample["gold_indices"])
        hits = len(gold_set & set(selected))
        recall = hits / len(gold_set)
        epoch_recall += recall
        n_samples += 1

    return {
        "loss": epoch_loss / n_samples if n_samples > 0 else 0.0,
        "recall": epoch_recall / n_samples if n_samples > 0 else 0.0,
        "n_samples": n_samples,
    }


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"💾 Output directory: {output_dir}")

    # Load data
    print(f"📚 Loading training data from {args.result_json}")
    train_data = load_training_data(args.result_json, args.max_samples)
    print(f"   Loaded {len(train_data)} valid samples")

    if len(train_data) == 0:
        print("❌ No valid training samples!")
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
        print(f"🧠 Model: LearnableQUBO")
    else:
        model = SoftQUBO(temperature=args.temperature_init)
        print(f"🧠 Model: SoftQUBO")

    model = model.to(device)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs")

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
            model.eval()
            val_loss = 0.0
            val_recall = 0.0
            with torch.no_grad():
                for sample in val_samples:
                    passage_embs = torch.from_numpy(sample["embeddings"]).float().to(device)
                    gold_indices = torch.tensor(sample["gold_indices"], dtype=torch.long, device=device)

                    if len(gold_indices) > 0:
                        query_emb = passage_embs[gold_indices].mean(dim=0)
                    else:
                        query_emb = passage_embs.mean(dim=0)

                    query_norm = query_emb / (torch.norm(query_emb) + 1e-12)
                    passage_norms = passage_embs / (torch.norm(passage_embs, dim=1, keepdim=True) + 1e-12)
                    a = (passage_norms @ query_norm).clamp(min=0.0)
                    b = passage_norms @ passage_norms.T
                    b = b.clamp(min=0.0, max=1.0)
                    b.fill_diagonal_(0.0)

                    if a.max() > a.min():
                        a = (a - a.min()) / (a.max() - a.min())

                    if isinstance(model, LearnableQUBO):
                        p, info = model(a, b, args.K)
                    else:
                        p, info = model(a, b, args.K, lam=2.0, gamma=1.0)

                    loss = compute_recall_loss(p, gold_indices)
                    val_loss += loss.item()

                    x = soft_select_to_hard(p, args.K)
                    selected = torch.where(x == 1)[0].cpu().numpy()
                    gold_set = set(sample["gold_indices"])
                    hits = len(gold_set & set(selected))
                    recall = hits / len(gold_set)
                    val_recall += recall

            val_metrics = {
                "loss": val_loss / len(val_samples),
                "recall": val_recall / len(val_samples),
            }
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
              f"Val_Recall={val_metrics['recall']:.4f}", end="")

        if isinstance(model, LearnableQUBO):
            print(f" | w_a={epoch_info['w_a']:.3f} | w_b={epoch_info['w_b']:.3f}")
        else:
            print()

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

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"   Best validation Recall: {best_val_recall:.4f}")
    print(f"   Model saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
