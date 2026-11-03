from glob import glob
import os
import pandas as pd
from pathlib import Path
import random
import stanza
# stanza.download("pt")
nlp = stanza.Pipeline("pt", download_method=None)


verbs_file = "./dataframes/verbs.csv"
verbs_df = pd.read_csv(verbs_file)

contractions_file = "./dataframes/contraction.csv"
contractions_df = pd.read_csv(contractions_file)

def process_contraction(token):
    text1 = token.words[0].text
    lemma1 = token.words[0].lemma
    
    candidates = contractions_df.loc[
        (contractions_df["lemma1"] == lemma1) &
        (contractions_df["text1"] != text1),
        "text"
    ].dropna().tolist()

    if not candidates:
        return token.text 
    
    rng = random.Random()
    new_word = rng.choice(candidates)
    return new_word

def process_verb(word):
    text = word.text
    lemma = word.lemma

    candidates = verbs_df.loc[
        (verbs_df["lemma"] == lemma) &
        (verbs_df["text"] != text),
        "text"
    ].dropna().tolist()

    if not candidates:
        return text

    rng = random.Random()
    new_word = rng.choice(candidates)
    return new_word

def mutate_token(token):
    words = token.words

    if len(words) > 1:
        return process_contraction(token)

    word = words[0]

    if word.pos == "PUNCT":
        return token.text

    if word.pos == "VERB":
        return process_verb(word)

    return token.text


def process_text(text):
    doc = nlp(text)

    pieces = []
    cursor = 0

    for sent in doc.sentences:
        for token in sent.tokens:
            start = token.start_char
            end = token.end_char

            pieces.append(text[cursor:start])
            pieces.append(mutate_token(token))

            cursor = end

    pieces.append(text[cursor:])

    return "".join(pieces)