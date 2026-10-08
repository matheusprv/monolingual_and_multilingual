"""Ponto de entrada dos experimentos, configurado exclusivamente por YAML.

Uso::

    python3 -m src.experiments.pipeline experiment.yaml causal
    python3 -m src.experiments.pipeline experiment.yaml predict
    python3 -m src.experiments.pipeline experiment.yaml train
    python3 -m src.experiments.pipeline experiment.yaml evaluate
    python3 -m src.experiments.pipeline experiment.yaml all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import set_seed

from . import hatebr_lora_causal_multimodel as experiment
from .causal_sliding import execute_experiment
from .dataset import load_causal_corpus, load_hatebr, make_stratified_splits, save_splits
from .experiment_config import ExperimentConfig, ModelSpec, load_experiment_config
from .model import LoRA_Model, Model
from .utils import clean_memory


def _model_dir(config: ExperimentConfig, model_spec: ModelSpec) -> Path:
    return config.output_dir / "models" / experiment.safe_model_name(model_spec.name)


def _result_dir(config: ExperimentConfig, model_spec: ModelSpec) -> Path:
    return config.output_dir / "results" / experiment.safe_model_name(model_spec.name)


def _save_progress(results: list[dict[str, Any]], config: ExperimentConfig) -> None:
    pd.DataFrame(results).to_csv(config.output_dir / "models_summary.csv", index=False)
    experiment.save_json({"results": results}, config.output_dir / "models_summary.json")


def _save_action_result(
    result: dict[str, Any], config: ExperimentConfig, model_spec: ModelSpec
) -> Path:
    """Salva uma linha de resultado por modelo e etapa, como no causal sliding."""
    output_dir = _result_dir(config, model_spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_row = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        for key, value in result.items()
    }
    output_path = output_dir / f"pipeline_{result['action']}.csv"
    pd.DataFrame([csv_row]).to_csv(output_path, index=False)
    return output_path


def _prepare_splits(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Carregando dataset e criando splits estratificados...")
    dataframe = load_hatebr(config.dataset.name)
    splits = make_stratified_splits(
        dataframe,
        test_size=config.dataset.test_size,
        validation_size=config.dataset.validation_size,
        seed=config.seed,
    )
    save_splits(splits, config.output_dir / "splits")
    return splits


def _prepare_causal_corpus(config: ExperimentConfig) -> pd.DataFrame:
    print(f"Lendo corpus causal: {config.causal.parquet_path}")
    return load_causal_corpus(
        config.causal.parquet_path,
        text_column=config.causal.text_column,
        scenarios=config.causal.scenarios,
    )


def _run_action(
    action: str,
    model_spec: ModelSpec,
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None,
    causal_corpus: pd.DataFrame | None,
    config: ExperimentConfig,
) -> dict[str, Any]:
    model_dir = _model_dir(config, model_spec)
    result_dir = _result_dir(config, model_spec)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    experiment.save_json(
        {
            "model_name": model_spec.name,
            "backend": model_spec.backend,
            "path": model_spec.path,
            "tokenizer": model_spec.tokenizer,
        },
        model_dir / "config.json",
    )

    if action == "all":
        zero_shot = _run_action("predict", model_spec, splits, causal_corpus, config)
        training = _run_action("train", model_spec, splits, causal_corpus, config)
        finetuned = _run_action("evaluate", model_spec, splits, causal_corpus, config)
        return {
            "model": model_spec.name,
            "status": "ok",
            "training_path": training["training_path"],
            "zero_shot_accuracy": zero_shot["accuracy"],
            "zero_shot_macro_f1": zero_shot["macro_f1"],
            "finetuned_accuracy": finetuned["accuracy"],
            "finetuned_macro_f1": finetuned["macro_f1"],
        }

    if action in {"predict", "train", "evaluate"} and splits is None:
        raise ValueError(f"A ação {action!r} requer os splits do dataset de classificação.")
    if action in {"causal", "evaluate"} and causal_corpus is None:
        raise ValueError(f"A ação {action!r} requer o corpus causal Parquet.")
    train_df, validation_df, test_df = splits if splits is not None else (None, None, None)

    if action == "train":
        model = Model(
            model_spec,
            config.model.use_4bit,
            native_training=model_spec.backend != "hf",
        )
        try:
            if model.supports_lora:
                training_path = experiment.train_lora(
                    model, train_df, validation_df, model_dir, config
                )
            else:
                training_path = experiment.train_native(
                    model, train_df, validation_df, model_dir, config
                )
            return {
                "model": model_spec.name,
                "status": "ok",
                "training_path": str(training_path),
            }
        finally:
            del model
            clean_memory()

    if action == "causal":
        file_name = experiment.safe_model_name(model_spec.name)
        results = execute_experiment(
            model_spec=model_spec,
            quantization=config.model.use_4bit,
            corpus=causal_corpus,
            output_folder=config.causal.results_dir,
            stride=config.causal.stride,
            max_len=config.causal.max_length,
            output_file_name=file_name,
        )
        if results is None:
            raise RuntimeError("A avaliação causal falhou; consulte o log acima.")
        return {
            "model": model_spec.name,
            "status": "ok",
            "rows": len(results),
            "causal_result_file": str(config.causal.results_dir / f"{file_name}.csv"),
        }

    if action == "predict":
        model = Model(model_spec, config.model.use_4bit)
        try:
            metrics = experiment.evaluate_classification(model, test_df, result_dir / "hatebr_zero_shot", config)
            return {"model": model_spec.name, "status": "ok", **metrics}
        finally:
            del model
            clean_memory()

    if model_spec.backend == "hf":
        training_path = model_dir / "hatebr_lora_adapter"
        if not training_path.exists():
            raise FileNotFoundError(f"Adapter não encontrado em {training_path}. Execute 'train' antes de 'evaluate'.")
        model = LoRA_Model(model_spec, config.model.use_4bit, str(training_path))
    else:
        training_path = model_dir / "hatebr_full_finetuned.pt"
        if not training_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint Candeia treinado não encontrado em {training_path}. "
                "Execute 'train' antes de 'evaluate'."
            )
        model = Model(
            model_spec,
            config.model.use_4bit,
            native_weights_path=training_path,
        )
    try:
        metrics = experiment.evaluate_classification(model, test_df, result_dir / "hatebr_finetuned", config)
        experiment.run_eval_causal_sliding(model, causal_corpus, result_dir, config)
        return {"model": model_spec.name, "status": "ok", "training_path": str(training_path), **metrics}
    finally:
        del model
        clean_memory()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pipeline modular dos experimentos HateBR.")
    parser.add_argument("config", type=Path, help="Arquivo YAML com toda a configuração do experimento.")
    parser.add_argument("action", choices=("causal", "predict", "train", "evaluate", "all"))
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    experiment.save_json(config.as_dict(), config.output_dir / "experiment_config.json")
    set_seed(config.seed)
    needs_splits = args.action in {"predict", "train", "evaluate", "all"}
    needs_causal_corpus = args.action in {"causal", "evaluate", "all"}
    splits = _prepare_splits(config) if needs_splits else None
    causal_corpus = _prepare_causal_corpus(config) if needs_causal_corpus else None
    results: list[dict[str, Any]] = []
    for model_spec in config.models:
        try:
            result = _run_action(args.action, model_spec, splits, causal_corpus, config)
            result["action"] = args.action
        except Exception as error:
            clean_memory()
            result = {"model": model_spec.name, "action": args.action, "status": "error", "error": f"{type(error).__name__}: {error}"}
            experiment.save_json(result, _result_dir(config, model_spec) / f"{args.action}_error.json")
        result["result_file"] = str(_save_action_result(result, config, model_spec))
        results.append(result)
        _save_progress(results, config)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
