# Files structure

Given a source text in brazillian portuguese, this text will be submitted to multiple modifications
- **Translation**: The text will have its version both in English and Spanish
- **Random permutation**: The order of the words will be permuted, generating a non trivial text. 
    - Ex: "O carro é azul" -> "azul carro O é"
- **Corrupted verbs**: The verb's conjugation will be explicitly wrong 
    - Ex: "Ela gosta de carros" -> "Ela **gostavam** de carros"
- **Word-align**: The word-align allows to make syntatic modification in the structure of the text. Given the text in portuguese and in the target language, we can make two different modifications:
    - **Lexical Swap**: A sentence originally written in portuguese will keep its structure, but using the target language lexical information.
        * Original: A garota usa um vestido rosa
        * Lexical Swap: The girl wears a dress pink
    - **Grammar Swap**: A sentence originally written in portuguse will kee its lexical and will start using the target language grammar.
        * Original: A garota usa um vestido rosa
        * Grammar Swap: A garota usa um rosa vestido



# Obtaining data
1. The brazillian portuguese text were obtained from the dataset [Corpus PT-BR v1](https://huggingface.co/datasets/bratao/corpus-ptbr-v1), Where 500 texts with 140 <= `num_of_words` <= 150 were selected.
2. These texts were then translated into english using `facebook/nllb-200-distilled-600M`
3. With the translated texts, it is possible to generate the alignment using the ubisoft-laforge-binary-align algorithm; Then, it is possible to generate both lexical and grammar swap


# Files
- Source
    - [portuguese_texts.txt](portuguese_texts.txt): Contain the portuguese base texts
- Translation:
    - [translation](translation/translation.json): the json contains the text from portuguese, english and spanish translations
- Grammar and lexical swap
    - [alignments.json](word_align/alignments.json): Is the result of the word alignment from ubisoft-laforge
    - [alignment_reconstruction.json](word_align/alignment_reconstruction.json): Using the indexes gave by the word-align, reconstruct the sentences changing the words structure, generating grammar swap and lexical swap
