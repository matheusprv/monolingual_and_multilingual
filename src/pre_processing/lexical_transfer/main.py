import stanza
import pandas as pd
from prompt import build_prompt
from model import Model
import os
import traceback
from tqdm import tqdm

os.makedirs("prompts/input", exist_ok=True)
os.makedirs("prompts/output", exist_ok=True)

MODEL_NAME = "../../models/models/google/gemma-4-31B-it/"

model = Model(MODEL_NAME, device_map="cuda:0")


LANGUAGE_TO_STANZA = {
    "INGLÊS": "en",
    "RUSSO": "ru",
    "FRANCÊS": "fr",
    "ESPANHOL": "es",
    "ÁRABE": "ar",
    "GALEGO": "gl",
    "MANDARIM": "zh-hans",
}



en = stanza.Pipeline(
    lang="en",
    processors="tokenize",
    download_method=None,
)
es = stanza.Pipeline(
    lang="es",
    processors="tokenize",
    download_method=None,
)
fr = stanza.Pipeline(
    lang="fr",
    processors="tokenize",
    download_method=None,
)

nlps = {
    "INGLÊS": en,
    "ESPANHOL": es,
    "FRANCÊS": fr
}


def tokenize(text, language):
    doc = nlps[language](text)

    tokens = []

    for sentence in doc.sentences:
        for token in sentence.tokens:
            tokens.append(token.text)

    return tokens


df = pd.read_parquet("TRANSLATION.parquet")

df = df[df["language"].isin(["INGLÊS", "ESPANHOL", "FRANCÊS"])]
df["text_id"] = df.index
df = df.reset_index(drop=True)


results = []

for index, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Processando textos"
):

    # Valores padrão caso alguma etapa falhe
    tokens = None
    prompt = None
    response = None
    error = None
    error_type = None

    try:
        tokens = tokenize(row["text"], row["language"])

        prompt = build_prompt(row["text"], tokens, row["language"])

        with open(f"prompts/input/{index}.txt", "w", encoding="utf-8") as file:
            file.write(prompt)

        response = model.message(prompt)
        with open(f"prompts/output/{index}.txt", "w", encoding="utf-8") as file:
            file.write(response)

    except Exception as e:
        error = str(e)
        error_type = type(e).__name__

        print(
            f"[ERROR] index={index} "
            f"text_id={row['text_id']} "
            f"{error_type}: {error}"
        )

        # Opcional: salva o traceback completo para debug
        with open(f"prompts/output/{index}_error.txt", "w", encoding="utf-8") as file:
            file.write(traceback.format_exc())

    results.append({
        "text_id": row["text_id"],
        "scenario": row["scenario"],
        "language": row["language"],
        "task": row["task"],
        "task_type": row["task_type"],
        "text": row["text"],
        "stanza_tokens": tokens,
        "model_output": response,
        "error_type": error_type,
        "error": error,
    })


results_df = pd.DataFrame(results)

results_df.to_parquet(
    "LEXICAL_SWAP.parquet",
    index=False
)
