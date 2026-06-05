
from glob import glob
import stanza
import pandas as pd
# stanza.download("pt")
nlp = stanza.Pipeline("pt", download_method=None)


def process_verb(verb, verbs_df):    
    new_row = {
        "text": verb.text,
        "lemma": verb.lemma
    }

    if verb.feats is None:
        return verbs_df
    
    feats_dict = dict(feat.split("=") for feat in verb.feats.split("|"))

    new_row = new_row | feats_dict
    df_row = pd.DataFrame([new_row])

    verbs_df = pd.concat([verbs_df, df_row], ignore_index=True)

    return verbs_df


def process_contraction(contraction, contraction_df):
    new_row = {
        "text": contraction.text
    }

    for i, word in enumerate(contraction.words):
        new_row[f"text{i+1}"] = word.text
        new_row[f"lemma{i+1}"] = word.lemma
    
    df_row = pd.DataFrame([new_row])
    contraction_df = pd.concat([contraction_df, df_row], ignore_index=True)

    return contraction_df


def extract_text_info(text, contraction_df=None, verbs_df=None):
    if contraction_df is None:
        contraction_df = pd.DataFrame()
    if verbs_df is None:
        verbs_df = pd.DataFrame()

    doc = nlp(text)
    print(text)
    for sent in doc.sentences:
        for token in sent.tokens:
            # Check if word is a contraction
            if len(token.words) > 1:
                contraction_df = process_contraction(token, contraction_df)
            else:
                word = token.words[0]
                if word.pos == "VERB":
                    verbs_df = process_verb(word, verbs_df)

    contraction_df.drop_duplicates(inplace=True)
    verbs_df.drop_duplicates(inplace=True)

    return contraction_df, verbs_df



def try_reading_dataframe(file_name):
    try:
        return pd.read_csv(file_name)
    except:
        return None

def process_corpus(file_paths):
    contraction_df = try_reading_dataframe("contraction.csv")
    verbs_df = try_reading_dataframe("verbs.csv")
    
    for i, file_path in enumerate(file_paths):
        
        print(f"File {i+1}/{len(file_paths)} - {file_path}")

        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                contraction_df, verbs_df = extract_text_info(line, contraction_df, verbs_df)
    
    contraction_df.to_csv("contraction.csv", index=False)
    verbs_df.to_csv("verbs.csv", index=False)

    return contraction_df, verbs_df


corpora = glob("../../../corpora/Universal Dependencies/*/*.txt", recursive=True)
corpora = [x for x in corpora if 'LICENSE' not in x]
corpora

process_corpus(corpora)


