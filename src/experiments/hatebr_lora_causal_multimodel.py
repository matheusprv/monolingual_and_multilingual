"""
Fine-tuning de Causal LM para classificação no HateBR usando LoRA,
preservando a capacidade generativa por não trocar a cabeça do modelo.

Saídas geradas em OUTPUT_DIR:
- hatebr_zero_shot_metrics.json
- hatebr_zero_shot_predictions.csv
- hatebr_lora_adapter/                  # modelo LoRA salvo
- hatebr_finetuned_metrics.json
- hatebr_finetuned_predictions.csv
- hatebr_eval_causal_sliding.csv
- hatebr_eval_causal_sliding_summary.json

Instalação sugerida:
pip install -U torch transformers datasets accelerate peft bitsandbytes scikit-learn pandas tqdm

O carregamento do modelo e a liberação de memória são centralizados em
`model.py` e `utils.py`. Este arquivo fica responsável pelo pipeline do HateBR.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)
import inspect

from .causal_sliding import eval_causal_sliding
from .model import LoRA_Model, Model
from .utils import clean_memory


# =========================
# Configuração principal
# =========================
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
]
MODELS = [
    m.strip()
    for m in os.getenv("MODELS", ",".join(DEFAULT_MODELS)).split(",")
    if m.strip()
]
DATASET_NAME = os.getenv("DATASET_NAME", "franciellevargas/HateBR")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs_hatebr_lora"))
SEED = int(os.getenv("SEED", "42"))

USE_4BIT = os.getenv("USE_4BIT", "1") == "1"
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
EVAL_CAUSAL_MAX_ROWS = int(os.getenv("EVAL_CAUSAL_MAX_ROWS", "0"))  # 0 = test inteiro

# Hiperparâmetros LoRA/treino.
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
NUM_EPOCHS = float(os.getenv("NUM_EPOCHS", "3"))
LR = float(os.getenv("LR", "2e-4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "4"))

ID2LABEL = {0: "não ofensivo", 1: "ofensivo"}
def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_prompt(text: str) -> str:
    return (
        "Classifique o comentário abaixo em exatamente uma das classes: "
        "ofensivo ou não ofensivo.\n\n"
        f"Comentário: {text}\n\n"
        "Classe:"
    )


def normalize_prediction(generated: str) -> int:
    s = generated.strip().lower()
    s = re.sub(r"[^a-záàâãéêíóôõúçãõ\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Ordem importa: "não ofensivo" contém "ofensivo".
    if "não ofensivo" in s or "nao ofensivo" in s or s.startswith("não") or s.startswith("nao"):
        return 0
    if "ofensivo" in s:
        return 1

    # Fallback conservador: se o modelo não seguiu o formato, marca como ofensivo
    # apenas quando há sinal textual explícito; caso contrário não ofensivo.
    return 0


def dataset_to_df(dataset_name: str) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    df = ds.to_pandas()

    if "comentario" not in df.columns or "label_final" not in df.columns:
        raise ValueError(
            f"Esperava colunas 'comentario' e 'label_final'. Colunas encontradas: {list(df.columns)}"
        )

    df = df[["comentario", "label_final"]].rename(columns={"comentario": "text", "label_final": "label"})
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df.dropna(subset=["text", "label"]).reset_index(drop=True)


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
def predict_generate(model: Model, texts: list[str]) -> list[int]:
    model.set_evaluation_mode()
    language_model = model.model
    tokenizer = model.tokenizer
    preds = []
    device = language_model.get_input_embeddings().weight.device

    for text in tqdm(texts, desc="Classificando", unit="ex"):
        prompt = build_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
        out = language_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
        preds.append(normalize_prediction(generated))

    return preds


def evaluate_classification(model: Model, df: pd.DataFrame, out_prefix: Path) -> dict[str, Any]:
    y_true = df["label"].astype(int).tolist()
    y_pred = predict_generate(model, df["text"].tolist())

    pred_df = df.copy()
    pred_df["prediction"] = y_pred
    pred_df["label_name"] = pred_df["label"].map(ID2LABEL)
    pred_df["prediction_name"] = pred_df["prediction"].map(ID2LABEL)
    pred_df.to_csv(out_prefix.parent / f"{out_prefix.name}_predictions.csv", index=False)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "f1_ofensivo": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=[ID2LABEL[0], ID2LABEL[1]],
            zero_division=0,
            output_dict=True,
        ),
    }
    save_json(metrics, out_prefix.parent / f"{out_prefix.name}_metrics.json")
    return metrics


# =========================
# Dataset causal para SFT
# =========================
def encode_example(example: dict[str, Any], tokenizer) -> dict[str, list[int]]:
    prompt = build_prompt(example["text"])
    answer = " " + ID2LABEL[int(example["label"])] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids

    input_ids = (prompt_ids + answer_ids)[:MAX_LENGTH]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_LENGTH]
    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


@dataclass
class CausalCollator:
    tokenizer: Any

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
        attention_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def train_lora(
    base_model: Model,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    tokenizer = base_model.tokenizer
    language_model = base_model.model

    if USE_4BIT and torch.cuda.is_available():
        language_model = prepare_model_for_kbit_training(language_model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=infer_lora_targets(language_model),
    )
    tuned_model = get_peft_model(language_model, lora_config)
    tuned_model.print_trainable_parameters()

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)
    train_ds = train_ds.map(lambda x: encode_example(x, tokenizer), remove_columns=train_ds.column_names)
    val_ds = val_ds.map(lambda x: encode_example(x, tokenizer), remove_columns=val_ds.column_names)

    # Compatibilidade entre versões do transformers: algumas usam
    # evaluation_strategy, outras aceitam eval_strategy.
    strategy_arg = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    args_kwargs = dict(
        output_dir=str(output_dir / "trainer_checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
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
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train()

    adapter_dir = output_dir / "hatebr_lora_adapter"
    tuned_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return adapter_dir


# =========================
# PPL / BPB com eval_causal_sliding
# =========================
def run_eval_causal_sliding(
    model: Model,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    corpus = test_df[["text"]].copy().reset_index(drop=True)
    if EVAL_CAUSAL_MAX_ROWS > 0:
        corpus = corpus.head(EVAL_CAUSAL_MAX_ROWS)

    results = eval_causal_sliding(model=model, corpus=corpus)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "hatebr_eval_causal_sliding.csv", index=False)

    summary = {
        "rows": int(len(results_df)),
        "mean_ppl": float(results_df["ppl"].replace([np.inf, -np.inf], np.nan).mean()),
        "median_ppl": float(results_df["ppl"].replace([np.inf, -np.inf], np.nan).median()),
        "mean_bpb": float(results_df["bpb"].replace([np.inf, -np.inf], np.nan).mean()),
        "mean_average_nll": float(results_df["average_nll"].replace([np.inf, -np.inf], np.nan).mean()),
    }
    save_json(summary, output_dir / "hatebr_eval_causal_sliding_summary.json")
    return results_df


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_name).strip("_")


def prepare_splits(output_dir: Path):
    print("1) Carregando HateBR e criando split único para todos os modelos...")
    df = dataset_to_df(DATASET_NAME)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=SEED, stratify=train_df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "hatebr_train.csv", index=False)
    val_df.to_csv(splits_dir / "hatebr_val.csv", index=False)
    test_df.to_csv(splits_dir / "hatebr_test.csv", index=False)
    return train_df, val_df, test_df


def run_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    model_dir = OUTPUT_DIR / safe_model_name(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_json({"model_name": model_name}, model_dir / "config.json")

    print(f"\n{'=' * 80}\nMODELO: {model_name}\n{'=' * 80}")
    base_model = Model(model_name, USE_4BIT)

    print("2) Avaliação zero-shot...")
    zero = evaluate_classification(
        base_model, test_df, model_dir / "hatebr_zero_shot"
    )

    print("3) Fine-tuning LoRA...")
    adapter_dir = train_lora(base_model, train_df, val_df, model_dir)

    # O objeto usado no treino não é mantido para a avaliação. O adapter é
    # recarregado pelo mesmo caminho usado em produção/inferência.
    del base_model
    clean_memory()

    lora_model = LoRA_Model(model_name, USE_4BIT, str(adapter_dir))

    print("4) Avaliação pós fine-tuning...")
    ft = evaluate_classification(
        lora_model, test_df, model_dir / "hatebr_finetuned"
    )

    print("5) Perplexidade/BPB pós fine-tuning...")
    ppl_df = run_eval_causal_sliding(lora_model, test_df, model_dir)
    ppl = {
        "mean_ppl": float(ppl_df["ppl"].replace([np.inf, -np.inf], np.nan).mean()),
        "median_ppl": float(ppl_df["ppl"].replace([np.inf, -np.inf], np.nan).median()),
        "mean_bpb": float(ppl_df["bpb"].replace([np.inf, -np.inf], np.nan).mean()),
    }

    result = {
        "model": model_name,
        "status": "ok",
        "adapter_dir": str(adapter_dir),
        "zero_shot_accuracy": zero["accuracy"],
        "zero_shot_macro_f1": zero["macro_f1"],
        "finetuned_accuracy": ft["accuracy"],
        "finetuned_macro_f1": ft["macro_f1"],
        **ppl,
    }

    del lora_model
    clean_memory()
    return result


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = prepare_splits(OUTPUT_DIR)

    results = []
    for i, model_name in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Iniciando {model_name}")
        try:
            results.append(run_model(model_name, train_df, val_df, test_df))
        except Exception as e:
            clean_memory()
            result = {
                "model": model_name,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            }
            results.append(result)
            save_json(result, OUTPUT_DIR / safe_model_name(model_name) / "error.json")
            print(f"ERRO em {model_name}: {result['error']}")

        pd.DataFrame(results).to_csv(OUTPUT_DIR / "models_summary.csv", index=False)
        save_json({"results": results}, OUTPUT_DIR / "models_summary.json")

    print("\nConcluído.")
    print(pd.DataFrame(results).to_string(index=False))
    print("Resultados em:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
