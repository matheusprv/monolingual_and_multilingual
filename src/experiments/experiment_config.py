"""Schema e leitura da configuração YAML dos experimentos."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    test_size: float
    validation_size: float


@dataclass(frozen=True)
class ModelConfig:
    use_4bit: bool


@dataclass(frozen=True)
class GenerationConfig:
    max_length: int
    max_new_tokens: int


@dataclass(frozen=True)
class TrainingConfig:
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    num_epochs: float
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int


@dataclass(frozen=True)
class CausalConfig:
    parquet_path: Path
    results_dir: Path
    text_column: str
    scenarios: tuple[str, ...] | None
    max_rows: int
    stride: int
    max_length: int | None


@dataclass(frozen=True)
class ExperimentConfig:
    models: tuple[str, ...]
    output_dir: Path
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    generation: GenerationConfig
    training: TrainingConfig
    causal: CausalConfig

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["causal"]["parquet_path"] = str(self.causal.parquet_path)
        data["causal"]["results_dir"] = str(self.causal.results_dir)
        return data


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"A seção obrigatória {name!r} deve ser um mapa YAML.")
    return value


def _value(section: dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ValueError(f"A chave obrigatória {section_name}.{key} não foi informada.")
    return section[key]


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Lê e valida toda a configuração necessária para uma execução."""
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("O arquivo de configuração deve conter um mapa YAML.")

    dataset = _section(data, "dataset")
    model = _section(data, "model")
    generation = _section(data, "generation")
    training = _section(data, "training")
    causal = _section(data, "causal")
    models = data.get("models")
    if not isinstance(models, list) or not models or not all(isinstance(item, str) and item.strip() for item in models):
        raise ValueError("A chave obrigatória 'models' deve ser uma lista não vazia de IDs de modelo.")

    output_dir = Path(_value(data, "output_dir", "raiz"))
    scenarios = _value(causal, "scenarios", "causal")
    if scenarios != "all" and (not isinstance(scenarios, list) or not all(isinstance(item, str) for item in scenarios)):
        raise ValueError("causal.scenarios deve ser 'all' ou uma lista de cenários.")
    max_length = _value(causal, "max_length", "causal")
    return ExperimentConfig(
        models=tuple(models),
        output_dir=output_dir,
        seed=int(_value(data, "seed", "raiz")),
        dataset=DatasetConfig(
            name=str(_value(dataset, "name", "dataset")),
            test_size=float(_value(dataset, "test_size", "dataset")),
            validation_size=float(_value(dataset, "validation_size", "dataset")),
        ),
        model=ModelConfig(use_4bit=bool(_value(model, "use_4bit", "model"))),
        generation=GenerationConfig(
            max_length=int(_value(generation, "max_length", "generation")),
            max_new_tokens=int(_value(generation, "max_new_tokens", "generation")),
        ),
        training=TrainingConfig(
            lora_r=int(_value(training, "lora_r", "training")),
            lora_alpha=int(_value(training, "lora_alpha", "training")),
            lora_dropout=float(_value(training, "lora_dropout", "training")),
            num_epochs=float(_value(training, "num_epochs", "training")),
            learning_rate=float(_value(training, "learning_rate", "training")),
            batch_size=int(_value(training, "batch_size", "training")),
            gradient_accumulation_steps=int(_value(training, "gradient_accumulation_steps", "training")),
        ),
        causal=CausalConfig(
            parquet_path=Path(_value(causal, "parquet_path", "causal")),
            results_dir=Path(_value(causal, "results_dir", "causal")),
            text_column=str(_value(causal, "text_column", "causal")),
            scenarios=None if scenarios == "all" else tuple(scenarios),
            max_rows=int(_value(causal, "max_rows", "causal")),
            stride=int(_value(causal, "stride", "causal")),
            max_length=None if max_length is None else int(max_length),
        ),
    )
