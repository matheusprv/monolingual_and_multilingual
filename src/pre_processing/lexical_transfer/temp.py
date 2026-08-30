import stanza


stanza.Pipeline(
    lang="en",
    processors="tokenize",
)
stanza.Pipeline(
    lang="es",
    processors="tokenize",
)
stanza.Pipeline(
    lang="fr",
    processors="tokenize",
)
