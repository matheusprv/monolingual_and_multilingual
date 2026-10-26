"""Leitura e preparação dos datasets usados pelos experimentos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

HATEBR_TEXT_COLUMN = "comentario"
HATEBR_LABEL_COLUMN = "label_final"


@dataclass(frozen=True)
class ClassificationTask:
    """Metadados necessários para treinar e avaliar uma tarefa de classificação."""

    dataset_name: str
    text_column: str
    label_column: str
    labels: tuple[str, ...]
    example_name: str
    slug: str
    test_split: str | None = None


HATEBR_TASK = ClassificationTask(
    dataset_name="franciellevargas/HateBR",
    text_column=HATEBR_TEXT_COLUMN,
    label_column=HATEBR_LABEL_COLUMN,
    labels=("não ofensivo", "ofensivo"),
    example_name="comentário",
    slug="hatebr",
)

AG_NEWS_PT_TASK = ClassificationTask(
    dataset_name="maritaca-ai/ag_news_pt",
    text_column="text",
    label_column="label",
    labels=("Mundo", "Esportes", "Negócios", "Tecnologia"),
    example_name="notícia",
    slug="ag_news_pt",
    test_split="test",
)

_CLASSIFICATION_TASKS = {
    HATEBR_TASK.dataset_name: HATEBR_TASK,
    AG_NEWS_PT_TASK.dataset_name: AG_NEWS_PT_TASK,
}


def get_classification_task(dataset_name: str) -> ClassificationTask:
    """Retorna o esquema de uma tarefa de classificação suportada."""
    try:
        return _CLASSIFICATION_TASKS[dataset_name]
    except KeyError as error:
        supported = ", ".join(sorted(_CLASSIFICATION_TASKS))
        raise ValueError(
            f"Dataset de classificação não suportado: {dataset_name!r}. "
            f"Datasets disponíveis: {supported}."
        ) from error


def load_classification_dataset(dataset_name: str, *, split: str = "train", text_column: str = HATEBR_TEXT_COLUMN, label_column: str = HATEBR_LABEL_COLUMN) -> pd.DataFrame:
    """Carrega um split Hugging Face e o normaliza para as colunas ``text`` e ``label``."""
    dataframe = load_dataset(dataset_name, split=split).to_pandas()
    missing = {text_column, label_column}.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dataset {dataset_name!r} não possui as colunas esperadas: {sorted(missing)}. Colunas disponíveis: {list(dataframe.columns)}")
    dataframe = dataframe[[text_column, label_column]].dropna().rename(columns={text_column: "text", label_column: "label"})
    dataframe["text"] = dataframe["text"].astype(str).str.strip()
    dataframe = dataframe[dataframe["text"].ne("")]
    dataframe["label"] = pd.to_numeric(dataframe["label"], errors="raise").astype(int)
    return dataframe.reset_index(drop=True)


def load_hatebr(dataset_name: str = "franciellevargas/HateBR") -> pd.DataFrame:
    """Carrega o HateBR no formato interno de classificação."""
    return load_classification_dataset(dataset_name)


def load_task_dataset(task: ClassificationTask, *, split: str = "train") -> pd.DataFrame:
    """Carrega uma tarefa registrada e normaliza suas colunas internas."""
    dataframe = load_classification_dataset(
        task.dataset_name,
        split=split,
        text_column=task.text_column,
        label_column=task.label_column,
    )
    expected_labels = set(range(len(task.labels)))
    unexpected_labels = set(dataframe["label"].unique()).difference(expected_labels)
    if unexpected_labels:
        raise ValueError(
            f"O split {split!r} de {task.dataset_name!r} contém rótulos não mapeados: "
            f"{sorted(unexpected_labels)}. Esperados: {sorted(expected_labels)}."
        )
    return dataframe


def load_causal_corpus(
    parquet_path: Path,
    *,
    text_column: str = "text",
    scenarios: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Lê e valida o corpus Parquet usado para perplexidade e BPB."""
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Corpus Parquet não encontrado: {parquet_path}")
    dataframe = pd.read_parquet(parquet_path)
    if text_column not in dataframe.columns:
        raise ValueError(f"Coluna de texto {text_column!r} não encontrada em {parquet_path}. Colunas: {list(dataframe.columns)}")
    if scenarios is not None:
        if "scenario" not in dataframe.columns:
            raise ValueError("O filtro de cenários requer uma coluna 'scenario' no corpus.")
        dataframe = dataframe[dataframe["scenario"].isin(scenarios)]
    dataframe = dataframe.dropna(subset=[text_column]).copy()
    dataframe[text_column] = dataframe[text_column].astype(str).str.strip()
    dataframe = dataframe[dataframe[text_column].ne("")]
    if text_column != "text":
        dataframe = dataframe.rename(columns={text_column: "text"})
    return dataframe.reset_index(drop=True)


def make_stratified_splits(dataframe: pd.DataFrame, *, test_size: float, validation_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cria splits treino/validação/teste estratificados e com índices limpos."""
    train, test = train_test_split(dataframe, test_size=test_size, random_state=seed, stratify=dataframe["label"])
    train, validation = train_test_split(train, test_size=validation_size, random_state=seed, stratify=train["label"])
    return tuple(split.reset_index(drop=True) for split in (train, validation, test))


def make_stratified_validation_split(
    dataframe: pd.DataFrame, *, validation_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa treino e validação preservando a proporção de classes."""
    train, validation = train_test_split(
        dataframe,
        test_size=validation_size,
        random_state=seed,
        stratify=dataframe["label"],
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def save_splits(splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], output_dir: Path, *, prefix: str = "hatebr") -> None:
    """Persiste os splits efetivamente usados pelo experimento."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, dataframe in zip(("train", "val", "test"), splits, strict=True):
        dataframe.to_csv(output_dir / f"{prefix}_{name}.csv", index=False)
