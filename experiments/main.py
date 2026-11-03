import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from causal_sliding import eval_causal_sliding
from model import LoRAModel, Model


def read_config(config_path):
    with config_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(path, base_dir):
    path = Path(path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_model_path(path, base_dir):
    if path.startswith((".", "/")):
        return str(resolve_path(path, base_dir))
    return path


def get_experiment_configs(config):
    keys = ("experiments", "dataset", "output_dir")
    values = [config.get(key) for key in keys]

    if not all(isinstance(value, list) for value in values):
        raise ValueError(f"As chaves {keys} devem ser listas.")
    if len({len(value) for value in values}) != 1:
        raise ValueError(f"As listas {keys} devem ter o mesmo tamanho.")

    return zip(*values)


def load_model(model_config, config_dir):
    backend = model_config["backend"]
    model_path = resolve_model_path(model_config["path"], config_dir)
    tokenizer_path = model_config.get("tokenizer")
    if tokenizer_path is not None:
        tokenizer_path = resolve_model_path(tokenizer_path, config_dir)
    quantization = model_config.get("quantization", False)

    print(f"⏳ Loading model {model_config['name']}")


    if backend == "hf":
        return Model(model_path, tokenizer_path, quantization)
    if backend == "peft":
        adapter_path = resolve_model_path(
            model_config.get("adapter_path", model_config["path"]),
            config_dir,
        )
        base_model_path = model_config.get("base_model_path")
        if base_model_path is not None:
            base_model_path = resolve_model_path(base_model_path, config_dir)
        return LoRAModel(
            adapter_path,
            tokenizer_path,
            quantization,
            base_model_path,
        )

    raise ValueError(f"Backend não suportado: {backend!r}")


def output_filename(model_name):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    return f"{safe_name or 'model'}.csv"


def run_causal_sliding(model, dataset, model_name, batch_size, max_length=None):
    required_columns = {
        "id",
        "text",
        "scenario",
        "language",
        "original_text_id",
    }
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"O dataset não possui as colunas necessárias: {missing_columns}"
        )

    result_frames = []
    for scenario, scenario_data in dataset.groupby("scenario", sort=False):
        print(f"Scenario: {scenario}")
        texts = scenario_data["text"].fillna("").astype(str).tolist()
        results = eval_causal_sliding(
            model, texts, batch_size=batch_size, max_len=max_length,
        )
        result = pd.DataFrame(results)
        result["text_id"] = scenario_data["id"].tolist()
        result.insert(0, "original_text_id", scenario_data["original_text_id"].tolist())
        result.insert(0, "language", scenario_data["language"].tolist())
        result.insert(0, "scenario", scenario)
        result.insert(0, "model", model_name)
        result_frames.append(result)

    return pd.concat(result_frames, ignore_index=True)


def run_experiment(experiment, dataset_path, output_dir, model_configs, config_dir):
    if experiment != "causal_sliding":
        raise ValueError(f"Experimento não suportado: {experiment!r}")

    dataset = pd.read_parquet(dataset_path, engine="fastparquet")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_config in model_configs:
        model = load_model(model_config, config_dir)
        results = run_causal_sliding(
            model,
            dataset,
            model_config["name"],
            batch_size=model_config.get("batch_size", 1),
            max_length=model_config.get("max_length"),
        )
        results.to_csv(
            output_dir / output_filename(model_config["name"]),
            index=False,
        )
        del model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("experiment.yaml"),
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = read_config(config_path)
    model_configs = config.get("models", [])
    if not model_configs:
        raise ValueError("A chave 'models' deve conter pelo menos um modelo.")

    for experiment, dataset, output_dir in get_experiment_configs(config):
        run_experiment(
            experiment,
            resolve_path(dataset, config_path.parent),
            resolve_path(output_dir, config_path.parent),
            model_configs,
            config_path.parent,
        )


if __name__ == "__main__":
    main()
