import json
from pathlib import Path

from align import reconstruct_sentences


BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "alinhamentos.json"
OUTPUT_FILE = BASE_DIR / "teste.json"


def invert_alignment(alignment_pt_en):
    """Converte pares portugues-ingles para ingles-portugues."""
    return " ".join(
        f"{english_index}-{portuguese_index}"
        for pair in alignment_pt_en.split()
        for portuguese_index, english_index in [pair.split("-")]
    )


def main():
    with INPUT_FILE.open(encoding="utf-8") as file:
        records = json.load(file)

    aligned_records = []
    for record in records:
        alignment_en_pt = invert_alignment(record["alignment_pt_en"])
        aligned_records.append(
            {
                **record,
                "alignment_en_pt": alignment_en_pt,
                "aligned_sentences": reconstruct_sentences(
                    record["english"],
                    record["portuguese"],
                    alignment_en_pt,
                ),
            }
        )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(aligned_records, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
