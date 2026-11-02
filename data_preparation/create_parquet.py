import json
from pathlib import Path

import pandas as pd
from IPython.display import display


BASE_DIR = Path(__file__).parent
PORTUGUESE_FILE = BASE_DIR / "portuguese_texts.txt"
PERMUTATION_FILE = BASE_DIR / "permutation" / "permuted_portuguese_texts.txt"
GRAMMAR_SWAP_FILE = BASE_DIR / "word_align" / "grammar_swap.txt"
LEXICAL_SWAP_FILE = BASE_DIR / "word_align" / "lexical_swap.txt"
TRANSLATION_FILE = BASE_DIR / "translation" / "translation.json"
OUTPUT_FILE = BASE_DIR / "text_scenarios.parquet"


def read_lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def validate_lengths(columns):
    lengths = {name: len(texts) for name, texts in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"As fontes possuem quantidades diferentes de textos: {lengths}")


def main():
    translations = json.loads(TRANSLATION_FILE.read_text(encoding="utf-8"))
    columns = {
        "portuguese": read_lines(PORTUGUESE_FILE),
        "permutation": read_lines(PERMUTATION_FILE),
        "grammar_swap": read_lines(GRAMMAR_SWAP_FILE),
        "lexical_swap": read_lines(LEXICAL_SWAP_FILE),
        "english": [translation["english"] for translation in translations],
    }

    validate_lengths(columns)
    dataframe = pd.DataFrame(columns)

    display(dataframe)
    dataframe.to_parquet(OUTPUT_FILE, engine="fastparquet", index=False)


if __name__ == "__main__":
    main()
