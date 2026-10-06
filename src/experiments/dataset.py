"""Leitura e preparação dos datasets usados pelos experimentos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

HATEBR_TEXT_COLUMN = "comentario"
HATEBR_LABEL_COLUMN = "label_final"


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


def make_stratified_splits(dataframe: pd.DataFrame, *, test_size: float, validation_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cria splits treino/validação/teste estratificados e com índices limpos."""
    train, test = train_test_split(dataframe, test_size=test_size, random_state=seed, stratify=dataframe["label"])
    train, validation = train_test_split(train, test_size=validation_size, random_state=seed, stratify=train["label"])
    return tuple(split.reset_index(drop=True) for split in (train, validation, test))


def save_splits(splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], output_dir: Path, *, prefix: str = "hatebr") -> None:
    """Persiste os splits efetivamente usados pelo experimento."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, dataframe in zip(("train", "val", "test"), splits, strict=True):
        dataframe.to_csv(output_dir / f"{prefix}_{name}.csv", index=False)
