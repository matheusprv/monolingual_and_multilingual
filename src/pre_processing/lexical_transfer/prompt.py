import json

def build_prompt(
    text: str,
    tokens: list[str],
    language: str,
) -> str:
    """
    Constrói o prompt usado para a substituição lexical.

    O modelo recebe:
        - sentença completa;
        - língua;
        - tokens numerados.

    A numeração é essencial para permitir MWEs descontínuas.
    """

    indexed_tokens = [
        {
            "index": i,
            "token": token,
        }
        for i, token in enumerate(tokens)
    ]

    schema_example = {
        "units": [
            {
                "source_indices": [0],
                "source_tokens": ["The"],
                "target_tokens": ["O"],
            },
            {
                "source_indices": [1],
                "source_tokens": ["red"],
                "target_tokens": ["vermelho"],
            },
            {
                "source_indices": [2],
                "source_tokens": ["car"],
                "target_tokens": ["carro"],
            },
        ]
    }

    prompt = f"""
You are performing CONTEXTUAL LEXICAL TRANSFER from {language} into
Brazilian Portuguese (PT-BR).

This is NOT normal sentence translation.

Your job is to replace the lexical items of the SOURCE sentence with
their contextually correct Portuguese equivalents while preserving the
SOURCE language's word order and syntactic arrangement as much as
possible.

SOURCE SENTENCE:
{text}

SOURCE TOKENS WITH IMMUTABLE INDICES:
{json.dumps(indexed_tokens, ensure_ascii=False)}

IMPORTANT RULES:

1. DO NOT translate the sentence into natural Portuguese syntax.

Example:

Source:
"The red car"

Correct:
"O vermelho carro"

Incorrect:
"O carro vermelho"

The incorrect version changes the English constituent order into
Portuguese order, which is forbidden.


2. Translation MUST be contextual.

Do not translate words independently when their meaning depends on the
sentence.

For example, in:

"make new decisions"

"make" should correspond to "tomar", because "make decisions" means
"tomar decisões" in this context.


3. Detect MULTI-WORD EXPRESSIONS.

Multiple source tokens may correspond to a single Portuguese lexical
unit or expression.

This includes, but is not limited to:

- phrasal verbs
- idioms
- compound lexical items
- light-verb constructions
- fixed expressions
- collocations whose lexical meaning cannot be translated independently
- grammaticalized lexical constructions


4. Multi-word expressions MAY BE DISCONTINUOUS.

For example:

"Turn the TV off"

Tokens:
0 Turn
1 the
2 TV
3 off

"Turn ... off" is one phrasal verb.

A valid analysis can therefore contain:

{{
    "source_indices": [0, 3],
    "source_tokens": ["Turn", "off"],
    "target_tokens": ["Desligue"]
}}

The words between these indices must still be processed normally.

Do NOT translate "Turn" and "off" separately if they jointly express
the phrasal verb.


5. Every source token index MUST occur exactly once in the analysis.

If indices [0, 3] belong to the same lexical unit, neither index 0 nor
index 3 may occur in another unit.


6. Preserve punctuation.

Punctuation must be represented as a unit and translated to itself.

Example:

{{
    "source_indices": [3],
    "source_tokens": [","],
    "target_tokens": [","]
}}


7. Portuguese words should have the morphology appropriate to the
context whenever it can be inferred.

For example:

"The pretty girl"

should yield lexical equivalents corresponding to:

["A", "bonita", "menina"]

not:

["O", "bonito", "menina"]


8. Do NOT insert Portuguese words simply to make the result
grammatically natural.

Do not rearrange constituents.
Do not repair foreign syntax.
Do not paraphrase the sentence.


9. A source unit may produce:

- one Portuguese token;
- multiple Portuguese tokens;
- zero Portuguese tokens, if the source element is absorbed by a
  multi-word lexical correspondence.

However, prefer explicit multi-token groups rather than arbitrary
deletion.


10. source_tokens MUST exactly reproduce the tokens at source_indices.

Do not alter source_tokens.


11. Return ONLY valid JSON.

No explanation.
No Markdown.
No ```json fences.
No comments.

The required JSON schema is:

{json.dumps(schema_example, ensure_ascii=False, indent=2)}

Now analyze the actual SOURCE SENTENCE and return the JSON.
""".strip()

    return prompt
