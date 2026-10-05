Modelos instruction-tuned possuem perplexidades e BPB maiores do que suas versões base. Por isso serão deixados de lado para as análise sseguintes

| Modelo     | Mean base |       Mean -it |          Ganho Mean |         Ganho Mean (%) | Std base |        Std -it |           Ganho Std |        Ganho Std (%) |
| ---------- | --------: | -------------: | ------------------: | ---------------------: | -------: | -------------: | ------------------: | -------------------: |
| Qwen3.5-9B |     32.28 |          36.45 |           **+4.17** |            **+12.92%** |    51.72 |          59.66 |           **+7.94** |          **+15.35%** |
| Gemma4-12B |     32.28 | 740,685,076.01 | **+740,685,043.73** | **+2,294,563,332.50%** |    46.91 | 450,280,364.29 | **+450,280,317.38** | **+959,881,299.04%** |
| Tucano-2b4 |     58.89 |          61.73 |           **+2.84** |             **+4.82%** |    96.53 |         103.21 |           **+6.68** |           **+6.92%** |



As seguintes análises foram feitas com os textos somente em português ou inglês, para permitir uma comparação entre todas as sentenças

# 1 - Em média, quais cenários são mais diferentes para os modelos?

Tradução possui menor perplexidade - Esperado, pela presença dos modelos multilíngues



| scenario        |   count |     mean |       std |     min |     25% |      50% |      75% |       max |
|:----------------|--------:|---------:|----------:|--------:|--------:|---------:|---------:|----------:|
| TRANSLATION     |     492 |  15.6353 |   8.31725 | 4.1045  |  9.5218 |  13.7659 |  19.6417 |   72.8087 |
| PB NATIVO       |     492 |  21.1117 |  15.498   | 2.23695 | 10.7529 |  16.8736 |  26.775  |  119.373  |
| GRAMMAR SWAP v3 |     492 |  29.2648 |  19.2644  | 3.16403 | 14.2473 |  24.5278 |  37.9699 |  123.684  |
| LEXICAL SWAP    |     492 |  35.1075 |  27.7074  | 6.8164  | 18.5771 |  27.8136 |  42.8781 |  266.046  |
| PB CORRUPTED    |     492 |  70.9482 |  64.9959  | 6.57921 | 29.5783 |  51.4314 |  87.7439 |  549.361  |
| PB PERMUTED     |     492 | 263.132  | 249.511   | 6.85655 | 82.9209 | 182.839  | 346.155  | 1669.6    |

Tabela ordenada pela média - Somente os textos em português e inglês


# 2 - O que o tamanho do texto tem relação com a perplexidade e o BPB?


# 3 - Existe algum modelo que se manteve mais constante ao longo dos cenários?


# 4 - Os modelos dependem mais do léxico ou da gramática no inglês?




----

Fazer gráfico de linha mostrando o desempenho dos modelos ao longo dos experimentos (ordenar do menor BPB para o maior)