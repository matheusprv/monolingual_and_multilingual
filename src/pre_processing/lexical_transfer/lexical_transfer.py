# pip install -U pandas pyarrow tqdm stanza transformers accelerate torch

import json
from pathlib import Path

import pandas as pd
import stanza
import torch
from tqdm import tqdm

from model import Model
from prompt import build_prompt


# ============================================================
# CONFIGURAÇÃO
# ============================================================

INPUT_FILE = "TRANSLATION.parquet"
OUTPUT_FILE = "TRANSLATION_LEXICAL_PTBR.parquet"
CHECKPOINT_FILE = "TRANSLATION_LEXICAL_PTBR.checkpoint.jsonl"

MODEL_NAME = "../../models/models/google/gemma-4-12B-it/"

TEXT_COLUMN = "text"
LANGUAGE_COLUMN = "language"

SAVE_EVERY = 25


# ============================================================
# MAPEAMENTO DAS LÍNGUAS
# ============================================================

LANGUAGE_TO_STANZA = {
    "INGLÊS": "en",
    "RUSSO": "ru",
    "FRANCÊS": "fr",
    "ESPANHOL": "es",
    "ÁRABE": "ar",
    "GALEGO": "gl",
    "MANDARIM": "zh-hans",
}


# ============================================================
# STANZA
# ============================================================

def load_stanza_pipelines(use_gpu=False):

    pipelines = {}

    for language_name, lang_code in LANGUAGE_TO_STANZA.items():

        print(f"Loading Stanza: {language_name} -> {lang_code}")

        pipelines[language_name] = stanza.Pipeline(
            lang=lang_code,
            processors="tokenize",
            use_gpu=use_gpu,
            download_method=None,
            verbose=False,
            model_dir="/media/data/matheusvieira/stanza",
        )

    return pipelines


def stanza_tokenize(
    text: str,
    language: str,
    pipelines,
) -> list[str]:

    if language not in pipelines:
        raise ValueError(
            f"Língua não suportada: {language}"
        )

    doc = pipelines[language](text)

    tokens = []

    for sentence in doc.sentences:
        for token in sentence.tokens:
            tokens.append(token.text)

    return tokens


# ============================================================
# GEMMA
# ============================================================

def lexical_transfer(
    model: Model,
    text: str,
    language: str,
    source_tokens: list[str],
) -> str:
    """
    Envia o prompt ao Gemma e retorna a resposta exatamente
    como foi produzida pelo modelo.

    Nenhum parsing ou validação é realizado.
    """

    prompt = build_prompt(
        text=text,
        tokens=source_tokens,
        language=language,
    )

    return model.message(prompt)


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(
    checkpoint_path: str,
) -> dict[int, dict]:

    path = Path(checkpoint_path)

    if not path.exists():
        return {}

    results = {}

    with path.open("r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            results[item["__row_index__"]] = item

    return results


def append_checkpoint(
    checkpoint_path: str,
    result: dict,
):

    with open(
        checkpoint_path,
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                result,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# RESULTADOS -> DATAFRAME
# ============================================================

def results_to_dataframe(
    df: pd.DataFrame,
    results: dict[int, dict],
) -> pd.DataFrame:

    rows = []

    for row_index, row in df.iterrows():

        if row_index not in results:
            continue

        result = results[row_index]

        # Mantém todas as colunas originais
        output = row.to_dict()

        output.update(
            {
                "stanza_tokens": result["stanza_tokens"],
                "model_output": result["model_output"],
                "processing_error": result["processing_error"],
            }
        )

        rows.append(output)

    return pd.DataFrame(rows)


# ============================================================
# PIPELINE
# ============================================================

def process_dataset(
    input_file=INPUT_FILE,
    output_file=OUTPUT_FILE,
    checkpoint_file=CHECKPOINT_FILE,
    model_name=MODEL_NAME,
    use_gpu_stanza=False,
):

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print(f"Loading dataset: {input_file}")

    df = pd.read_parquet(input_file)

    # Cria text_id antes do filtro.
    # Assim text_id representa a posição original no dataframe.
    df = df.reset_index(drop=True)
    df["text_id"] = df.index

    # Por enquanto, somente inglês e espanhol
    df = df[
        df["language"].isin(
            [
                "INGLÊS",
                # "ESPANHOL",
            ]
        )
    ].copy()

    # O índice interno usado pelo checkpoint pode ser sequencial.
    # text_id continua sendo o ID do dataframe original.
    df = df.reset_index(drop=True)

    # df = df[:3]

    required_columns = {
        TEXT_COLUMN,
        LANGUAGE_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Colunas ausentes: {missing_columns}"
        )

    print(f"Rows: {len(df):,}")

    print(
        "Languages:",
        df[LANGUAGE_COLUMN]
        .value_counts()
        .to_dict(),
    )

    # --------------------------------------------------------
    # Stanza
    # --------------------------------------------------------

    print("\nLoading Stanza pipelines...")

    stanza_pipelines = load_stanza_pipelines(
        use_gpu=use_gpu_stanza
    )

    # --------------------------------------------------------
    # Gemma
    # --------------------------------------------------------

    print(f"\nLoading Gemma: {model_name}")

    model = Model(
        model_name=model_name,
        device_map="cuda",
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    results = load_checkpoint(
        checkpoint_file
    )

    if results:
        print(
            f"Checkpoint found: "
            f"{len(results):,} rows already processed."
        )

    processed_since_save = 0

    # --------------------------------------------------------
    # Processamento
    # --------------------------------------------------------

    progress = tqdm(
        df.iterrows(),
        total=len(df),
        desc="Lexical transfer",
        unit="text",
    )

    for row_index, row in progress:

        # Já foi processado anteriormente
        if row_index in results:
            continue

        text = row[TEXT_COLUMN]
        language = row[LANGUAGE_COLUMN]

        # ----------------------------------------------------
        # Texto vazio
        # ----------------------------------------------------

        if pd.isna(text) or not str(text).strip():

            result = {
                "__row_index__": row_index,
                "text_id": int(row["text_id"]),
                "stanza_tokens": [],
                "model_output": None,
                "processing_error": "EMPTY_TEXT",
            }

            results[row_index] = result

            append_checkpoint(
                checkpoint_file,
                result,
            )

            continue

        text = str(text)

        # ----------------------------------------------------
        # Inferência
        # ----------------------------------------------------

        source_tokens = None

        try:

            source_tokens = stanza_tokenize(
                text=text,
                language=language,
                pipelines=stanza_pipelines,
            )

            model_output = lexical_transfer(
                model=model,
                text=text,
                language=language,
                source_tokens=source_tokens,
            )

            result = {
                "__row_index__": row_index,
                "text_id": int(row["text_id"]),
                "stanza_tokens": source_tokens,

                # Salva exatamente o que o Gemma respondeu
                "model_output": model_output,

                "processing_error": None,
            }

        except Exception as exc:

            error_message = repr(exc)


            is_oom = (
                isinstance(exc, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(exc).lower()
                or "cuda oom" in str(exc).lower()
            )

            if is_oom:

                print(
                    "\n"
                    + "=" * 80
                )
                print("CUDA OUT OF MEMORY")
                print(f"row_index: {row_index}")
                print(f"text_id: {int(row['text_id'])}")
                print(f"language: {language}")
                print(f"text_length: {len(text):,} chars")
                print(f"error: {error_message}")
                print("=" * 80,flush=True,)

                # Tenta liberar memória não utilizada
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            result = {
                "__row_index__": row_index,
                "text_id": int(row["text_id"]),

                "stanza_tokens": source_tokens,

                "model_output": None,

                "processing_error": (
                    f"OOM: {error_message}"
                    if is_oom
                    else error_message
                ),
            }

        # ----------------------------------------------------
        # Checkpoint imediato
        # ----------------------------------------------------

        results[row_index] = result

        append_checkpoint(
            checkpoint_file,
            result,
        )

        processed_since_save += 1

        # ----------------------------------------------------
        # Salva parquet intermediário
        # ----------------------------------------------------

        if processed_since_save >= SAVE_EVERY:

            partial_df = results_to_dataframe(
                df,
                results,
            )

            partial_df.to_parquet(
                output_file,
                index=False,
            )

            processed_since_save = 0

    # --------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------

    output_df = results_to_dataframe(
        df,
        results,
    )

    output_df.to_parquet(
        output_file,
        index=False,
    )

    print("\nFinished.")
    print(f"Output: {output_file}")

    error_count = (
        output_df["processing_error"]
        .notna()
        .sum()
    )

    print(
        f"Successful: "
        f"{len(output_df) - error_count:,}"
    )

    print(
        f"Errors: "
        f"{error_count:,}"
    )

    return output_df


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    output_df = process_dataset(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        checkpoint_file=CHECKPOINT_FILE,
        model_name=MODEL_NAME,
        use_gpu_stanza=False,
    )