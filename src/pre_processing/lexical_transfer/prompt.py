import json

LANGUAGES = {
    "INGLÊS": "english",
    "ESPANHOL": "spanish",
    "FRANCÊS": "french"
}

def build_prompt(
    text: str,
    tokens: list[str],
    language: str,
) -> str:

    indexed_tokens = [
        [i, token]
        for i, token in enumerate(tokens)
    ]

    language = LANGUAGES[language]

    return f"""
Perform contextual lexical substitution from {language} to Brazilian Portuguese.

This is NOT normal sentence translation.


Replace lexical items of the SOURCE sentence with their contextually correct Portuguese equivalents while preserving the SOURCE language's word order and syntactic arrangement as much as possible.


Rules:
- Do not translate into natural Portuguese syntax. Example: "The red car" "The red car" -> "O vermelho carro", not "O carro vermelho"; "The beautfiul girl" -> "A bonita menina", not "A menina bonita"
- Translation must be contextual.
- Detect MULTI-WORD EXPRESSIONS, including discontinuous ones. Example: "Turn the TV off" -> 0,3=Desligue|1=a|2=TV
- The words between these indices must still be processed normally.
- Every source token index MUST occur exactly once in the analysis. If indices [0, 3] belong to the same lexical unit, neither index 0 nor index 3 may occur in another unit.
- Preserve punctuation
- Use contextually correct Portuguese morphology.
- Do NOT insert or remove Portuguese words simply to make the result grammatically natural.
- A surce unit may produce: one Portuguese token; multiple Portuguese tokens; zero Portuguese tokens, if the source element is absorbed by a multi-word lexical correspondence.
- Prefer explicit multi-token groups rather than arbitrary deletion.
- Every source index must appear exactly once.
- If multiple source tokens form one lexical unit, join their indices with commas.
- If one source unit translates to multiple Portuguese words, write them normally separated by spaces.
- The output must be the indices of the source words and the textual translation.

Sentence: {text}

Tokens: {json.dumps(indexed_tokens, ensure_ascii=False)}

Return ONLY this format:
index=translation|index=translation|...

Examples:
0=O|1=vermelho|2=carro

For a multi-word expression:
0,3=Desligue|1=a|2=TV
""".strip()