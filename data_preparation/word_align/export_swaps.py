import json
from pathlib import Path


BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "alignment_reconstruction.json"
LEXICAL_SWAP_FILE = BASE_DIR / "lexical_swap.txt"
GRAMMAR_SWAP_FILE = BASE_DIR / "grammar_swap.txt"


def main():
    with INPUT_FILE.open(encoding="utf-8") as file:
        records = json.load(file)

    lexical_swaps = [
        record["aligned_sentences"]["src_to_tgt"]["sentence"]
        for record in records
    ]
    grammar_swaps = [
        record["aligned_sentences"]["tgt_to_src"]["sentence"]
        for record in records
    ]

    LEXICAL_SWAP_FILE.write_text("\n".join(lexical_swaps), encoding="utf-8")
    GRAMMAR_SWAP_FILE.write_text("\n".join(grammar_swaps), encoding="utf-8")


if __name__ == "__main__":
    main()
