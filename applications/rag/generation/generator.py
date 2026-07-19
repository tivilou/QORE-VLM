"""LLM generation wrapper for RAG evaluation.

Unified interface for answer generation from selected passages. Handles prompt
construction (with chat template support), batching, and device management.
"""

from __future__ import annotations

from typing import Optional

import torch


class Generator:
    """LLM-based answer generator.

    Wraps a HuggingFace causal LM and tokenizer. Constructs prompts via the
    tokenizer's chat template when available (recommended for instruction-tuned
    models like Llama-3-Instruct), falling back to plain concatenation otherwise.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        torch_dtype=torch.float16,
        max_new_tokens: int = 32,
        use_chat_template: bool = True,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device or "auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.use_chat_template = use_chat_template

    def generate(self, question: str, passages: list[str]) -> str:
        """Generate an answer from the question and selected passages."""
        prompt = self._build_prompt(question, passages)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the NEW tokens (skip the prompt).
        input_len = inputs["input_ids"].shape[1]
        answer_ids = outputs[0][input_len:]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
        return answer.strip()

    def _build_prompt(self, question: str, passages: list[str]) -> str:
        context = "\n\n".join(
            f"Passage {i+1}: {p}" for i, p in enumerate(passages)
        )
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers questions "
                        "based on provided passages. Answer with only the specific "
                        "information requested - typically just a few words or a short phrase. "
                        "Do not add explanations or extra context."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a concise answer (typically 1-5 words):",
                },
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback if chat template fails
                pass

        # Plain concatenation fallback
        return (
            f"Answer the question based on the passages below. "
            f"Provide only a short, direct answer without explanation.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Short Answer:"
        )
