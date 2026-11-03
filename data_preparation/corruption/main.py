from pathlib import Path
from tqdm import tqdm
from corruption import process_text


BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR.parent / "portuguese_texts.txt"
OUTPUT_FILE = BASE_DIR / "portuguese_corrupted.txt"


def main():
    with INPUT_FILE.open(encoding="utf-8") as file:
        texts = file.read().splitlines()

    corrupted_texts = []
    for text in tqdm(texts):
        corrupted = process_text(text) 
        corrupted_texts.append(corrupted)
        
        
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        file.write("\n".join(corrupted_texts))


if __name__ == "__main__":
    main()
