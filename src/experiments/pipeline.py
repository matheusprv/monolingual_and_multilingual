"""Ponto de entrada para executar etapas isoladas dos experimentos.

- causal: PPL/BPB do modelo base
- predict: classificação zero-shot
- train: fine-tuning LoRA
- evaluate: classificação e PPL/BPB usando o adapter LoRA salvo
- all: fluxo completo anterior

Uso::

    python -m src.experiments.pipeline causal
    python -m src.experiments.pipeline predict
    python -m src.experiments.pipeline train
    python -m src.experiments.pipeline evaluate
    python -m src.experiments.pipeline all

As variáveis de ambiente do experimento original continuam disponíveis; por
exemplo ``MODELS``, ``OUTPUT_DIR`` e ``USE_4BIT``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import set_seed

from . import hatebr_lora_causal_multimodel as experiment
from .model import LoRA_Model, Model
from .utils import clean_memory


def _save_progress(results: list[dict[str, Any]]) -> None:
    pd.DataFrame(results).to_csv(experiment.OUTPUT_DIR / "models_summary.csv", index=False)
    experiment.save_json({"results": results}, experiment.OUTPUT_DIR / "models_summary.json")


def _save_action_result(result: dict[str, Any]) -> Path:
    """Salva o resultado de uma etapa no mesmo padrão por-modelo do causal.

    Diferentemente do resumo agregado, este arquivo representa somente uma
    execução (modelo + etapa). Estruturas como o relatório de classificação
    são serializadas como JSON para permanecerem legíveis e não quebrarem as
    colunas do CSV.
    """
    action = result["action"]
    model_name = result["model"]
    output_dir = experiment.OUTPUT_DIR / "results" / experiment.safe_model_name(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_row = {
        key: json.dumps(value, ensure_ascii=False)
        if isinstance(value, (dict, list)) else value
        for key, value in result.items()
    }
    output_path = output_dir / f"pipeline_{action}.csv"
    pd.DataFrame([csv_row]).to_csv(output_path, index=False)
    return output_path


def _run_action(action: str, model_name: str, splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> dict[str, Any]:
    train_df, validation_df, test_df = splits
    model_dir = experiment.OUTPUT_DIR / "models" / experiment.safe_model_name(model_name)
    result_dir = experiment.OUTPUT_DIR / "results" / experiment.safe_model_name(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    experiment.save_json({"model_name": model_name}, model_dir / "config.json")

    if action == "all":
        # Cada etapa usa seus diretórios próprios: modelos em ``models/`` e
        # resultados em ``results/``. Isso evita misturar checkpoints e CSVs.
        zero_shot = _run_action("predict", model_name, splits)
        training = _run_action("train", model_name, splits)
        finetuned = _run_action("evaluate", model_name, splits)
        return {
            "model": model_name,
            "status": "ok",
            "adapter_dir": training["adapter_dir"],
            "zero_shot_accuracy": zero_shot["accuracy"],
            "zero_shot_macro_f1": zero_shot["macro_f1"],
            "finetuned_accuracy": finetuned["accuracy"],
            "finetuned_macro_f1": finetuned["macro_f1"],
        }

    if action == "train":
        model = Model(model_name, experiment.USE_4BIT)
        try:
            adapter_dir = experiment.train_lora(model, train_df, validation_df, model_dir)
            return {"model": model_name, "status": "ok", "adapter_dir": str(adapter_dir)}
        finally:
            del model
            clean_memory()

    if action in {"predict", "causal"}:
        model = Model(model_name, experiment.USE_4BIT)
        try:
            if action == "predict":
                metrics = experiment.evaluate_classification(model, test_df, result_dir / "hatebr_zero_shot")
                return {"model": model_name, "status": "ok", **metrics}
            experiment.run_eval_causal_sliding(model, test_df, result_dir)
            return {"model": model_name, "status": "ok"}
        finally:
            del model
            clean_memory()

    # A avaliação pós fine-tuning recarrega o adapter salvo. Assim ela pode
    # ser executada em outro processo ou após uma interrupção no treinamento.
    adapter_dir = model_dir / "hatebr_lora_adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter não encontrado em {adapter_dir}. Execute 'train' antes de 'evaluate'.")
    model = LoRA_Model(model_name, experiment.USE_4BIT, str(adapter_dir))
    try:
        metrics = experiment.evaluate_classification(model, test_df, result_dir / "hatebr_finetuned")
        experiment.run_eval_causal_sliding(model, test_df, result_dir)
        return {"model": model_name, "status": "ok", "adapter_dir": str(adapter_dir), **metrics}
    finally:
        del model
        clean_memory()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pipeline modular dos experimentos HateBR.")
    parser.add_argument("action", choices=("causal", "predict", "train", "evaluate", "all"))
    parser.add_argument("--models", nargs="+", default=None, help="IDs Hugging Face; substitui MODELS.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Substitui OUTPUT_DIR.")
    args = parser.parse_args(argv)

    if args.models:
        experiment.MODELS = args.models
    if args.output_dir:
        experiment.OUTPUT_DIR = args.output_dir
    experiment.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(experiment.SEED)
    splits = experiment.prepare_splits(experiment.OUTPUT_DIR)
    results: list[dict[str, Any]] = []
    for model_name in experiment.MODELS:
        try:
            result = _run_action(args.action, model_name, splits)
            result["action"] = args.action
        except Exception as error:
            clean_memory()
            result = {"model": model_name, "action": args.action, "status": "error", "error": f"{type(error).__name__}: {error}"}
            error_dir = experiment.OUTPUT_DIR / "results" / experiment.safe_model_name(model_name)
            experiment.save_json(result, error_dir / f"{args.action}_error.json")
        result["result_file"] = str(_save_action_result(result))
        results.append(result)
        _save_progress(results)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
