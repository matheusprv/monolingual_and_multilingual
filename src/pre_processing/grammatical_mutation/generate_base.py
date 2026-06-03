import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import stanza

from tqdm import tqdm

def parse_feats(feats: str | None) -> dict:
    """
    Converte os traços morfológicos do Stanza/UD em um dicionário.

    Exemplo:
        "Gender=Masc|Number=Sing"
    vira:
        {"Gender": "Masc", "Number": "Sing"}
    """
    if not feats:
        return {}

    parsed = {}

    for item in feats.split("|"):
        if "=" in item:
            key, value = item.split("=", 1)
            parsed[key] = value

    return parsed


def read_text_file(path: str | Path) -> str:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    return path.read_text(encoding="utf-8")


def split_text_into_chunks(text: str, max_chars: int = 20_000) -> list[str]:
    """
    Divide textos grandes em blocos menores para evitar problemas de memória.

    A divisão é feita preferencialmente por parágrafos.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current_chunk = []

    current_size = 0

    for paragraph in paragraphs:
        paragraph_size = len(paragraph)

        if current_size + paragraph_size > max_chars and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(paragraph)
        current_size += paragraph_size

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def build_stanza_pipeline(use_gpu: bool = False):
    """
    Cria o pipeline do Stanza para português.

    Processadores usados:
    - tokenize: tokenização
    - mwt: expansão de multi-word tokens
    - pos: classe gramatical e traços morfológicos
    - lemma: lema
    - depparse: dependências sintáticas

    O depparse não é obrigatório para conjugação, mas pode ser útil depois
    para corromper concordância com base em relações sintáticas.
    """
    return stanza.Pipeline(
        lang="pt",
        processors="tokenize,mwt,pos,lemma,depparse",
        use_gpu=use_gpu,
        tokenize_no_ssplit=False,
        verbose=False,
    )


def process_text(text: str, nlp, source_name: str = "input") -> pd.DataFrame:
    rows = []

    chunks = split_text_into_chunks(text)

    global_sentence_id = 0
    global_word_id = 0

    for chunk_id, chunk in enumerate(
        tqdm(chunks, desc="Processando chunks", unit="chunk")
    ):
        doc = nlp(chunk)

        for sentence in doc.sentences:
            global_sentence_id += 1
            sentence_text = sentence.text

            for word in sentence.words:
                global_word_id += 1
                feats_dict = parse_feats(word.feats)

                row = {
                    "source": source_name,
                    "chunk_id": chunk_id,
                    "sentence_id": global_sentence_id,
                    "word_global_id": global_word_id,
                    "word_id": word.id,
                    "text": word.text,
                    "text_lower": word.text.lower(),
                    "lemma": word.lemma,
                    "lemma_lower": word.lemma.lower() if word.lemma else None,
                    "upos": word.upos,
                    "xpos": word.xpos,
                    "feats": word.feats,
                    "head": word.head,
                    "deprel": word.deprel,
                    "sentence_text": sentence_text,
                }

                row.update(feats_dict)
                rows.append(row)

    return pd.DataFrame(rows)


def save_outputs(df: pd.DataFrame, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base completa: uma linha por ocorrência de palavra
    df.to_csv(output_dir / "tokens_completo.csv", index=False)
    df.to_parquet(output_dir / "tokens_completo.parquet", index=False)

    # Base sem substantivos, útil para corrupção morfológica
    non_nouns = df[df["upos"] != "NOUN"].copy()
    non_nouns.to_csv(output_dir / "tokens_sem_substantivos.csv", index=False)
    non_nouns.to_parquet(output_dir / "tokens_sem_substantivos.parquet", index=False)

    # Bases separadas por classe gramatical universal
    upos_dir = output_dir / "por_upos"
    upos_dir.mkdir(exist_ok=True)

    for upos, group in df.groupby("upos"):
        safe_name = upos.lower()
        group.to_csv(upos_dir / f"{safe_name}.csv", index=False)
        group.to_parquet(upos_dir / f"{safe_name}.parquet", index=False)

    # Base lexical agregada:
    # uma linha por combinação única de forma, lema, UPOS e traços morfológicos
    group_cols = [
        "text_lower",
        "lemma_lower",
        "upos",
        "xpos",
        "feats",
    ]

    optional_feature_cols = [
        "Gender",
        "Number",
        "Person",
        "Tense",
        "Mood",
        "VerbForm",
        "Voice",
        "Aspect",
        "Definite",
        "PronType",
        "Poss",
        "Case",
        "Degree",
        "NumType",
    ]

    existing_feature_cols = [
        col for col in optional_feature_cols
        if col in df.columns
    ]

    group_cols = group_cols + existing_feature_cols

    lexicon = (
        df
        .groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="frequency")
        .sort_values(["lemma_lower", "upos", "frequency"], ascending=[True, True, False])
    )

    lexicon.to_csv(output_dir / "lexico_agregado.csv", index=False)
    lexicon.to_parquet(output_dir / "lexico_agregado.parquet", index=False)

    # Léxico separado por classe gramatical
    lexicon_dir = output_dir / "lexico_por_upos"
    lexicon_dir.mkdir(exist_ok=True)

    for upos, group in lexicon.groupby("upos"):
        safe_name = upos.lower()
        group.to_csv(lexicon_dir / f"{safe_name}.csv", index=False)
        group.to_parquet(lexicon_dir / f"{safe_name}.parquet", index=False)

    # Base específica para verbos
    if "VERB" in df["upos"].unique() or "AUX" in df["upos"].unique():
        verbs = df[df["upos"].isin(["VERB", "AUX"])].copy()

        verbs.to_csv(output_dir / "verbos_ocorrencias.csv", index=False)
        verbs.to_parquet(output_dir / "verbos_ocorrencias.parquet", index=False)

        verb_lexicon = lexicon[lexicon["upos"].isin(["VERB", "AUX"])].copy()
        verb_lexicon.to_csv(output_dir / "verbos_lexico.csv", index=False)
        verb_lexicon.to_parquet(output_dir / "verbos_lexico.parquet", index=False)

    # Base específica para determinantes/artigos
    if "DET" in df["upos"].unique():
        dets = df[df["upos"] == "DET"].copy()

        dets.to_csv(output_dir / "determinantes_ocorrencias.csv", index=False)
        dets.to_parquet(output_dir / "determinantes_ocorrencias.parquet", index=False)

        det_lexicon = lexicon[lexicon["upos"] == "DET"].copy()
        det_lexicon.to_csv(output_dir / "determinantes_lexico.csv", index=False)
        det_lexicon.to_parquet(output_dir / "determinantes_lexico.parquet", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Gera bases morfossintáticas para corrupção de textos em português usando Stanza."
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Caminho para o arquivo .txt de entrada."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="base_stanza",
        help="Diretório onde os arquivos serão salvos."
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Usa GPU, se disponível."
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)

    print(f"Lendo arquivo: {input_path}")
    text = read_text_file(input_path)

    print("Carregando pipeline do Stanza...")
    nlp = build_stanza_pipeline(use_gpu=args.gpu)

    print("Processando texto...")
    df = process_text(
        text=text,
        nlp=nlp,
        source_name=input_path.name,
    )

    print(f"Total de palavras processadas: {len(df)}")

    print(f"Salvando arquivos em: {output_dir}")
    save_outputs(df, output_dir)

    print("Finalizado.")


if __name__ == "__main__":
    main()