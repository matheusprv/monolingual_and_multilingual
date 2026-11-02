from pathlib import Path

from permutation import permut_text


BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR.parent / "portuguese_texts.txt"
OUTPUT_FILE = "permuted_portuguese_texts.txt"


def main():
    with INPUT_FILE.open(encoding="utf-8") as file:
        texts = file.read().splitlines()

    permuted_texts = [permut_text(text) for text in texts]

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        file.write("\n".join(permuted_texts))


if __name__ == "__main__":
    main()
