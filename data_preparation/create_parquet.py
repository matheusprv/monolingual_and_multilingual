import json
from pathlib import Path

import pandas as pd
from IPython.display import display


BASE_DIR = Path(__file__).parent
PORTUGUESE_FILE = BASE_DIR / "portuguese_texts.txt"
PERMUTATION_FILE = BASE_DIR / "permutation" / "permuted_portuguese_texts.txt"
CORRUPTION_FILE = BASE_DIR / "corruption" / "portuguese_corrupted.txt"
GRAMMAR_SWAP_FILE = BASE_DIR / "word_align" / "grammar_swap.txt"
LEXICAL_SWAP_FILE = BASE_DIR / "word_align" / "lexical_swap.txt"
TRANSLATION_FILE = BASE_DIR / "translation" / "translation.json"
OUTPUT_FILE = BASE_DIR / "text_scenarios.parquet"


def read_lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def validate_lengths(texts_by_scenario):
    lengths = {
        scenario: len(texts)
        for scenario, texts in texts_by_scenario.items()
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"As fontes possuem quantidades diferentes de textos: {lengths}")


def create_rows(texts_by_scenario):
    rows = []
    original_texts = texts_by_scenario["original"]

    for original_text_id, text in enumerate(original_texts):
        rows.append(
            {
                "id": original_text_id,
                "text": text,
                "scenario": "original",
                "language": "portuguese",
                "original_text_id": original_text_id,
            }
        )

    for scenario, language in (
        ("permutation", "portuguese"),
        ("corruption", "portuguese"),
        ("grammar_swap", "english"),
        ("lexical_swap", "english"),
        ("translation", "english"),
    ):
        for original_text_id, text in enumerate(texts_by_scenario[scenario]):
            rows.append(
                {
                    "id": len(rows),
                    "text": text,
                    "scenario": scenario,
                    "language": language,
                    "original_text_id": original_text_id,
                }
            )

    return rows


def main():
    translations = json.loads(TRANSLATION_FILE.read_text(encoding="utf-8"))
    texts_by_scenario = {
        "original": read_lines(PORTUGUESE_FILE),
        "permutation": read_lines(PERMUTATION_FILE),
        "corruption": read_lines(CORRUPTION_FILE),
        "grammar_swap": read_lines(GRAMMAR_SWAP_FILE),
        "lexical_swap": read_lines(LEXICAL_SWAP_FILE),
        "translation": [translation["english"] for translation in translations],
    }

    validate_lengths(texts_by_scenario)
    dataframe = pd.DataFrame(create_rows(texts_by_scenario))

    display(dataframe)
    dataframe.to_parquet(OUTPUT_FILE, engine="fastparquet", index=False)


if __name__ == "__main__":
    main()
