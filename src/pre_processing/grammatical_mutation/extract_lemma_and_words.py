from conllu import parse_incr
import json
import random
import nltk
from collections import defaultdict


def iter_conllu_sentences(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        yield from parse_incr(file)


def extract_corpora_data(corpora):
    """
        word_to_lemma and lemma_to_word
        word_to_lemma: 
            - a dict of each word in the exact way they appear on the text and its corresponding lemma
            - if the word is a contraction, the value will be the lemma of the first token, followed by '_prefix'
        
        lemma_to_words: 
            - A dict with all the lemmas and the words from its lemma
            - if the lemma is from a contraction it will have the following strucure:
                lemma1_prefix: [list of words with this lemma1]
    """
    word_to_lemma = dict()
    lemma_to_words = defaultdict(set)

    for corpus in corpora:
        print(corpus)
        
        for sentence in iter_conllu_sentences(corpus):
            num_tokens = len(sentence)

            i = 0
            while i < num_tokens:
                token = sentence[i]

                id_token = token["id"]
                form = token["form"]
                lemma = token["lemma"]
                lemma_to_words_key = lemma
                
                # Checking if the word is a contraction
                # If so, its lemma will be the lemma of the first word
                if not isinstance(id_token, int):
                    first_real_token = sentence[i+1]
                    lemma = first_real_token["lemma"] 
                    lemma_to_words_key = lemma + "_prefix"

                    i += id_token[-1] - id_token[0]

                word_to_lemma.setdefault(form, lemma)
                lemma_to_words[lemma_to_words_key].add(form.lower())
                                    

                i += 1
     
    lemma_to_words = {k: list(v) for k, v in lemma_to_words.items()}


    return word_to_lemma, lemma_to_words


corpora = [
    "../../../corpora/Universal Dependencies/UD_Portuguese-Bosque/pt_bosque-ud-dev.conllu",
    "../../../corpora/Universal Dependencies/UD_Portuguese-Bosque/pt_bosque-ud-test.conllu",
    "../../../corpora/Universal Dependencies/UD_Portuguese-Bosque/pt_bosque-ud-train.conllu",
    

    "../../../corpora/Universal Dependencies/UD_Portuguese-Porttinari/pt_porttinari-ud-dev.conllu",
    "../../../corpora/Universal Dependencies/UD_Portuguese-Porttinari/pt_porttinari-ud-test.conllu",
    "../../../corpora/Universal Dependencies/UD_Portuguese-Porttinari/pt_porttinari-ud-train.conllu"
]

word_to_lemma, lemma_to_words = extract_corpora_data(corpora)

with open("word_to_lemma.json", "w") as file:
    json.dump(word_to_lemma, file, indent=4, ensure_ascii=False)

with open("lemma_to_words.json", "w") as file:
    json.dump(lemma_to_words, file, indent=4, ensure_ascii=False)