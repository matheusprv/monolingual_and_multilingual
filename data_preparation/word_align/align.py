"""Reconstrução de frases a partir de alinhamentos no formato TALP.

Cada alinhamento TALP possui o formato ``indice_src-indice_tgt``.
"""


def reconstruct_sentences(src, tgt, talp):
    """Reconstrói frases nas duas direções e devolve os mapeamentos usados.

    ``src_to_tgt`` mantém a ordem de ``tgt`` e usa o léxico de ``src``.
    ``tgt_to_src`` mantém a ordem de ``src`` e usa o léxico de ``tgt``.

    Tokens sem alinhamento são ignorados. Quando um token é alinhado a mais de
    uma posição da outra frase, ele é inserido uma única vez na frase
    reconstruída. O campo ``word_mapping`` conserva todas as relações para
    conferência.
    """
    src_words = src.split()
    tgt_words = tgt.split()
    alignments = _parse_talp(talp, len(src_words), len(tgt_words))

    src_by_tgt = _group_alignments(alignments, key_index=1, value_index=0)
    tgt_by_src = _group_alignments(alignments, key_index=0, value_index=1)

    src_to_tgt_sentence, src_to_tgt_mapping = _reconstruct(
        structure_words=tgt_words,
        lexical_words=src_words,
        aligned_words=src_by_tgt,
        structure_name="tgt",
        lexical_name="src",
    )
    tgt_to_src_sentence, tgt_to_src_mapping = _reconstruct(
        structure_words=src_words,
        lexical_words=tgt_words,
        aligned_words=tgt_by_src,
        structure_name="src",
        lexical_name="tgt",
    )

    return {
        "src_to_tgt": {
            "sentence": src_to_tgt_sentence,
            "word_mapping": src_to_tgt_mapping,
        },
        "tgt_to_src": {
            "sentence": tgt_to_src_sentence,
            "word_mapping": tgt_to_src_mapping,
        },
    }


def _parse_talp(talp, src_size, tgt_size):
    alignments = []

    for pair in talp.split():
        try:
            src_index, tgt_index = (int(index) for index in pair.split("-"))
        except ValueError as error:
            raise ValueError(
                f"Alinhamento TALP inválido: {pair!r}. Use o formato 'src-tgt'."
            ) from error

        if not 0 <= src_index < src_size or not 0 <= tgt_index < tgt_size:
            raise ValueError(
                f"Alinhamento fora dos limites: {pair!r} "
                f"(src: 0-{src_size - 1}, tgt: 0-{tgt_size - 1})."
            )

        alignments.append((src_index, tgt_index))

    return alignments


def _group_alignments(alignments, key_index, value_index):
    grouped = {}
    for alignment in alignments:
        key = alignment[key_index]
        grouped.setdefault(key, []).append(alignment[value_index])
    return grouped


def _reconstruct(
    structure_words,
    lexical_words,
    aligned_words,
    structure_name,
    lexical_name,
):
    used_lexical_indices = set()
    reconstructed_words = []
    word_mapping = []

    for structure_index, structure_word in enumerate(structure_words):
        lexical_indices = aligned_words.get(structure_index, [])
        lexical_tokens = [lexical_words[index] for index in lexical_indices]
        emitted_indices = []
        emitted_tokens = set()

        for index in lexical_indices:
            lexical_word = lexical_words[index]
            if index in used_lexical_indices or lexical_word in emitted_tokens:
                continue
            emitted_indices.append(index)
            emitted_tokens.add(lexical_word)

        if emitted_indices:
            reconstructed_words.extend(lexical_words[index] for index in emitted_indices)
        used_lexical_indices.update(lexical_indices)

        word_mapping.append(
            {
                f"{structure_name}_index": structure_index,
                f"{structure_name}_word": structure_word,
                f"{lexical_name}_indices": lexical_indices,
                f"{lexical_name}_words": lexical_tokens,
                f"emitted_{lexical_name}_words": [
                    lexical_words[index] for index in emitted_indices
                ],
            }
        )

    return " ".join(reconstructed_words), word_mapping
