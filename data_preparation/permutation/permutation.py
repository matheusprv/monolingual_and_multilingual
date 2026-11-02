import random
import spacy
nlp = spacy.load("pt_core_news_lg")


SPACE_AFTER = {".", "..", "...", ",", ";", ":", "!", "?", "…", ")", "]", "}", "»", "”", "’", "%", "\""}
SPACE_BEFORE = {"(", "[", "{", "«", "“", "‘", "\""}


def permut_text(text: str):
    doc = nlp(text)

    chunk = []
    shuffled_text = ""

    total_tokens = len(doc)

    token = None
    for i in range(total_tokens):
        token = doc.__getitem__(i)

        if token.text in ('\n', '\n\n'):
            shuffled_text += token.text

        elif token.pos_ != "PUNCT":
            chunk.append(token.text)

        else:
            # Adding the shuffled text
            random.shuffle(chunk)
            shuffled_text += " ".join(chunk)

            chunk = [] 

            # Defining if there is a blank space before or after any punctuation
            if token.text in SPACE_BEFORE:
                if len(shuffled_text) == 0 or shuffled_text[-1] == " ":
                    shuffled_text += token.text
                else: 
                    shuffled_text += " " + token.text

            elif token.text in SPACE_AFTER:
                if i < total_tokens-1 and doc.__getitem__(i+1).pos_ == "PUNCT":
                    shuffled_text += token.text
                else: 
                    shuffled_text += token.text + " "
            else:
                shuffled_text += token.text
    
    # In case the last sentence does not end with a PUNCT
    if token.pos_ != "PUNCT":
        random.shuffle(chunk)
        shuffled_text += " ".join(chunk)

    return shuffled_text        

