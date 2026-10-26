"""
Fine-tuning de Causal LM para classificação textual usando LoRA,
preservando a capacidade generativa por não trocar a cabeça do modelo.

Saídas geradas no diretório de resultados configurado:
- <dataset>_zero_shot_metrics.json
- <dataset>_zero_shot_predictions.csv
- <dataset>_finetuned_metrics.json
- <dataset>_finetuned_predictions.csv
- <dataset>_eval_causal_sliding.csv
- <dataset>_eval_causal_sliding_summary.json

O adapter LoRA e os checkpoints são salvos no diretório de modelos configurado.

Instalação sugerida:
pip install -U torch transformers datasets accelerate peft bitsandbytes scikit-learn pandas tqdm

O carregamento do modelo e a liberação de memória são centralizados em
`model.py` e `utils.py`. A orquestração é feita por `pipeline.py`.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
)
import inspect

from .causal_sliding import eval_causal_sliding
from .dataset import ClassificationTask
from .experiment_config import ExperimentConfig
from .model import Model


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_prompt(text: str, task: ClassificationTask) -> str:
    classes = ", ".join(task.labels[:-1]) + f" ou {task.labels[-1]}"
    return (
        f"Classifique a {task.example_name} abaixo em exatamente uma das classes: "
        f"{classes}.\n\n"
        f"{task.example_name.capitalize()}: {text}\n\n"
        "Classe:"
    )


def _normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.casefold())).strip()


def normalize_prediction(generated: str, task: ClassificationTask) -> int:
    """Converte uma resposta gerada no índice da classe da tarefa.

    Os nomes mais longos são testados primeiro para evitar que uma classe que
    contém outra (por exemplo, ``não ofensivo``) seja interpretada incorretamente.
    """
    normalized = _normalized_text(generated)
    labels_by_length = sorted(
        enumerate(task.labels), key=lambda item: len(_normalized_text(item[1])), reverse=True
    )
    for label_id, label in labels_by_length:
        normalized_label = _normalized_text(label)
        if re.search(rf"(?:^|\s){re.escape(normalized_label)}(?:$|\s)", normalized):
            return label_id

    # Quando a geração não contém uma classe válida, a primeira classe mantém o
    # comportamento determinístico que o experimento já tinha no HateBR.
    return 0


def infer_lora_targets(model) -> list[str]:
    common = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "query_key_value", "dense", "fc1", "fc2",
    ]
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = [name for name in common if name in module_names]
    if not targets:
        raise ValueError("Não consegui inferir target_modules para LoRA neste modelo.")
    return targets


# =========================
# Avaliação de classificação
# =========================
@torch.inference_mode()
def predict_generate(
    model: Model, texts: list[str], config: ExperimentConfig, task: ClassificationTask
) -> list[int]:
    model.set_evaluation_mode()
    preds = []

    for text in tqdm(texts, desc="Classificando", unit="ex"):
        prompt = build_prompt(text, task)
        inputs = model.prepare_inputs(
            prompt, truncation=True, max_length=config.generation.max_length
        )
        out = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=config.generation.max_new_tokens,
        )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        generated = model.decode(gen_ids)
        preds.append(normalize_prediction(generated, task))

    return preds


def evaluate_classification(
    model: Model,
    df: pd.DataFrame,
    out_prefix: Path,
    config: ExperimentConfig,
    task: ClassificationTask,
) -> dict[str, Any]:
    y_true = df["label"].astype(int).tolist()
    y_pred = predict_generate(model, df["text"].tolist(), config, task)

    pred_df = df.copy()
    pred_df["prediction"] = y_pred
    id2label = dict(enumerate(task.labels))
    pred_df["label_name"] = pred_df["label"].map(id2label)
    pred_df["prediction_name"] = pred_df["prediction"].map(id2label)
    pred_df.to_csv(out_prefix.parent / f"{out_prefix.name}_predictions.csv", index=False)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    labels = list(range(len(task.labels)))
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    metrics: dict[str, Any] = {
        "dataset": task.dataset_name,
        "labels": list(task.labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "f1_por_classe": {label: float(score) for label, score in zip(task.labels, per_class_f1, strict=True)},
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=list(task.labels),
            zero_division=0,
            output_dict=True,
        ),
    }
    if task.labels == ("não ofensivo", "ofensivo"):
        metrics["f1_ofensivo"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    save_json(metrics, out_prefix.parent / f"{out_prefix.name}_metrics.json")
    return metrics


# =========================
# Dataset causal para SFT
# =========================
def encode_example(
    example: dict[str, Any], tokenizer, max_length: int, task: ClassificationTask
) -> dict[str, list[int]]:
    prompt = build_prompt(example["text"], task)
    answer = " " + task.labels[int(example["label"])] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids

    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
        attention_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def train_lora(
    base_model: Model,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
    config: ExperimentConfig,
    task: ClassificationTask,
) -> Path:
    if not base_model.supports_lora:
        raise ValueError("LoRA/PEFT só é suportado pelo backend 'hf'.")
    tokenizer = base_model.tokenizer
    language_model = base_model.model

    if config.model.use_4bit and torch.cuda.is_available():
        language_model = prepare_model_for_kbit_training(language_model)

    lora_config = LoraConfig(
        r=config.training.lora_r,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=infer_lora_targets(language_model),
    )
    tuned_model = get_peft_model(language_model, lora_config)
    tuned_model.print_trainable_parameters()

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)
    encode = lambda example: encode_example(example, tokenizer, config.generation.max_length, task)
    train_ds = train_ds.map(encode, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(encode, remove_columns=val_ds.column_names)

    # Compatibilidade entre versões do transformers: algumas usam
    # evaluation_strategy, outras aceitam eval_strategy.
    strategy_arg = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    args_kwargs = dict(
        output_dir=str(output_dir / "trainer_checkpoints"),
        num_train_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        save_strategy="epoch",
        logging_steps=25,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    args_kwargs[strategy_arg] = "epoch"
    args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=tuned_model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=CausalCollator(pad_token_id=base_model.pad_token_id),
    )
    trainer.train()

    adapter_dir = output_dir / f"{task.slug}_lora_adapter"
    tuned_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return adapter_dir


def encode_native_example(
    example: dict[str, Any], model: Model, max_length: int, task: ClassificationTask
) -> dict[str, list[int]] | None:
    """Prepara um exemplo SFT sem depender do tokenizer Transformers."""
    prompt_ids = model.encode_ids(build_prompt(str(example["text"]), task))
    answer_ids = model.encode_ids(" " + task.labels[int(example["label"])])
    if not answer_ids or answer_ids[-1] != model.eos_token_id:
        answer_ids.append(model.eos_token_id)
    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    # Sem ao menos uma predição supervisionada, a CE seria indefinida.
    if not any(token != -100 for token in labels[1:]):
        return None
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def _native_validation_loss(model: Model, loader: DataLoader) -> float:
    model.set_evaluation_mode()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(model.input_device)
            labels = batch["labels"].to(model.input_device)
            loss = model.native_causal_loss(input_ids, labels)
            tokens = int((labels[:, 1:] != -100).sum().item())
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens
    if total_tokens == 0:
        raise ValueError("A validação nativa não contém tokens supervisionados.")
    return total_loss / total_tokens


def train_native(
    base_model: Model,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
    config: ExperimentConfig,
    task: ClassificationTask,
) -> Path:
    """Ajuste fino completo para Candeia, sem PEFT/LoRA."""
    if base_model.supports_lora:
        raise ValueError("train_native é exclusivo dos backends Candeia.")

    encode = lambda row: encode_native_example(row, base_model, config.generation.max_length, task)
    train_examples = [encoded for row in train_df.to_dict("records") if (encoded := encode(row))]
    val_examples = [encoded for row in val_df.to_dict("records") if (encoded := encode(row))]
    if not train_examples or not val_examples:
        raise ValueError("Não restaram exemplos supervisionados após truncamento do prompt.")

    collator = CausalCollator(pad_token_id=base_model.pad_token_id)
    train_loader = DataLoader(
        train_examples,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_examples,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        base_model.model.parameters(), lr=config.training.learning_rate
    )
    accumulation_steps = max(1, config.training.gradient_accumulation_steps)
    checkpoint_path = output_dir / f"{task.slug}_full_finetuned.pt"
    best_validation_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(math.ceil(config.training.num_epochs)):
        base_model.set_train_mode()
        pending_steps = 0
        progress = tqdm(train_loader, desc=f"Treinando {base_model.spec.name} ({epoch + 1})", unit="batch")
        for batch in progress:
            input_ids = batch["input_ids"].to(base_model.input_device)
            labels = batch["labels"].to(base_model.input_device)
            loss = base_model.native_causal_loss(input_ids, labels)
            (loss / accumulation_steps).backward()
            pending_steps += 1
            if pending_steps == accumulation_steps:
                torch.nn.utils.clip_grad_norm_(base_model.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_steps = 0
            progress.set_postfix(loss=f"{loss.item():.4f}")
        if pending_steps:
            torch.nn.utils.clip_grad_norm_(base_model.model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        validation_loss = _native_validation_loss(base_model, val_loader)
        print(f"Epoch {epoch + 1}: validation loss={validation_loss:.6f}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            base_model.save_native_weights(checkpoint_path)

    if not checkpoint_path.is_file():
        raise RuntimeError("O treino nativo terminou sem produzir checkpoint.")
    save_json(
        {"best_validation_loss": best_validation_loss, "checkpoint": str(checkpoint_path)},
        output_dir / f"{task.slug}_full_finetuned_metrics.json",
    )
    return checkpoint_path


# =========================
# PPL / BPB com eval_causal_sliding
# =========================
def run_eval_causal_sliding(
    model: Model,
    corpus: pd.DataFrame,
    output_dir: Path,
    config: ExperimentConfig,
    task: ClassificationTask,
) -> pd.DataFrame:
    corpus = corpus.copy().reset_index(drop=True)
    if config.causal.max_rows > 0:
        corpus = corpus.head(config.causal.max_rows)

    results = eval_causal_sliding(
        model=model,
        corpus=corpus,
        stride=config.causal.stride,
        max_len=config.causal.max_length,
    )
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / f"{task.slug}_eval_causal_sliding.csv", index=False)

    summary = {
        "rows": int(len(results_df)),
        "mean_ppl": float(results_df["ppl"].replace([np.inf, -np.inf], np.nan).mean()),
        "median_ppl": float(results_df["ppl"].replace([np.inf, -np.inf], np.nan).median()),
        "mean_bpb": float(results_df["bpb"].replace([np.inf, -np.inf], np.nan).mean()),
        "mean_average_nll": float(results_df["average_nll"].replace([np.inf, -np.inf], np.nan).mean()),
    }
    save_json(summary, output_dir / f"{task.slug}_eval_causal_sliding_summary.json")
    return results_df


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_name).strip("_")
