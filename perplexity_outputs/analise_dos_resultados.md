Analisei a tabela usando **PB NATIVO como baseline por tarefa**. Em geral, valores negativos em `*_vs_PB_%` significam que o modelo teve **PPL menor que no PB nativo** naquela tarefa/língua; valores positivos significam **PPL maior**, ou seja, pior desempenho relativo ao PB. Os dados vêm da tabela enviada. 

## 1. Leitura geral dos resultados

O Francês é a língua com maior queda de perplexidade, em 25 dos casos


| Língua   | Média vs PB | Mediana vs PB | Nº de vezes melhor que PB |
| -------- | ----------: | ------------: | ------------------------: |
| FRANCÊS  |  **-17,6%** |    **-23,3%** |                 **25/32** |
| MANDARIM |        8,9% |         -1,3% |                     16/32 |
| INGLÊS   |       26,8% |          4,6% |                     12/32 |
| ESPANHOL |       65,5% |         21,3% |                     11/32 |
| GALEGO   |       84,9% |         89,7% |                      3/32 |
| RUSSO    |      101,0% |         53,8% |                      3/32 |
| ÁRABE    |  **165,2%** |    **134,7%** |                      6/32 |


**Interpretação:**
Espanhol e galego são mais próximos do português
Isso não gerou facilidades para o modelo
Russo e árabe foram os mais problemáticos

---

## 2. Por modelo

### Gemma

Médias por língua no Gemma:

| Língua   | Média vs PB |
| -------- | ----------: |
| FRANCÊS  |   **-5,7%** |
| INGLÊS   |        6,6% |
| MANDARIM |       12,9% |
| RUSSO    |       21,3% |
| ESPANHOL |       34,8% |
| GALEGO   |       98,6% |
| ÁRABE    |      152,7% |

Francês teve queda na perplexidade média

No **FAZER** o mandarim abaixou em 88,82% a perplexidade (249,30 -> 27,88)


---

### Gervásio

Gervásio teve média de **+58,1%** vs PB e apenas **11 de 56 comparações** melhores que PB.

| Língua   | Média vs PB |
| -------- | ----------: |
| FRANCÊS  |       10,7% |
| ESPANHOL |       13,1% |
| RUSSO    |       18,2% |
| MANDARIM |       22,9% |
| GALEGO   |       42,4% |
| INGLÊS   |      110,0% |
| ÁRABE    |      189,2% |

Francês e espanhol ficaram perto do português, mas o galego ficou longe
De certa forma, esperado, porque o modelo é monolinuge

Maior problema foi inglês e árabe

---

### Qwen

Qwen teve média de **+65,2%** vs PB, mas também muitos casos de melhora: **21 de 56 comparações** abaixo do PB.

| Língua   | Média vs PB |
| -------- | ----------: |
| FRANCÊS  |  **-37,0%** |
| MANDARIM |  **-30,6%** |
| INGLÊS   |       12,6% |
| ÁRABE    |       62,6% |
| RUSSO    |       85,5% |
| GALEGO   |      118,8% |
| ESPANHOL |  **244,8%** |

A perplexidade do Qwen é muito alta, ficando sempre na casa do milhar

Alta melhora no francês e mandarim

Qwen continua com crescimentos absurdos dos valores -> Isso pode ter refletido no esquecimento catastrofico do STS



---

### Tucano


| Língua   | Média vs PB |
| -------- | ----------: |
| FRANCÊS  |  **-38,4%** |
| ESPANHOL |  **-30,5%** |
| INGLÊS   |  **-21,9%** |
| MANDARIM |       30,4% |
| GALEGO   |       80,0% |
| ÁRABE    |      256,3% |
| RUSSO    |      279,1% |



O Tucano, focado em PTBR teve queda nas perplexidades para o frances e espanhol. 
Mas uma piora para o galego 

Porém apresentou os piores casos da tabela **RECOMENDAR/ÁRABE +516,08%**, **EXPLORAR/ÁRABE +488,70%** 

---

## 3. Por tarefa

A tarefa mais favorável foi **FAZER**. Ela teve média geral de **-25,1%** vs PB e **22 de 28 comparações** melhores que PB.

| Tarefa       | Média vs PB | Mediana vs PB | Nº de vezes melhor que PB |
| ------------ | ----------: | ------------: | ------------------------: |
| FAZER        |  **-25,1%** |    **-48,4%** |                 **22/28** |
| RELATAR      |       43,6% |         37,4% |                      9/28 |
| EXPLORAR     |       55,1% |         35,5% |                     11/28 |
| CAPACITAR    |       64,3% |         24,2% |                      8/28 |
| EXPLICAR     |       69,9% |         35,5% |                      8/28 |
| COMPARTILHAR |       83,5% |         58,4% |                      5/28 |
| RECRIAR      |       84,1% |         76,7% |                      8/28 |
| RECOMENDAR   |  **121,5%** |     **73,4%** |                      5/28 |

**Interpretação:**
A tarefa **FAZER** parece ser a mais “transferível” entre línguas. Já **RECOMENDAR** é a mais problemática, principalmente por causa de grandes aumentos em árabe, espanhol e galego dependendo do modelo.

---

## 4. Melhores e piores casos individuais

### Melhores casos

| Tarefa    | Língua   | Modelo | Variação vs PB |
| --------- | -------- | ------ | -------------: |
| FAZER     | MANDARIM | Gemma  |    **-88,82%** |
| CAPACITAR | MANDARIM | Qwen   |    **-83,99%** |
| FAZER     | FRANCÊS  | Tucano |    **-78,77%** |
| FAZER     | FRANCÊS  | Gemma  |    **-77,55%** |
| EXPLORAR  | INGLÊS   | Gemma  |    **-77,42%** |


### Piores casos

| Tarefa     | Língua   | Modelo   | Variação vs PB |
| ---------- | -------- | -------- | -------------: |
| RECOMENDAR | ÁRABE    | Tucano   |   **+516,08%** |
| EXPLORAR   | ÁRABE    | Tucano   |   **+488,70%** |
| RECOMENDAR | ÁRABE    | Gemma    |   **+442,79%** |
| RECOMENDAR | ÁRABE    | Gervásio |   **+418,11%** |
| RECOMENDAR | ESPANHOL | Qwen     |   **+398,17%** |


---


**Melhor língua geral:** francês.
**Língua mais problemática:** árabe.
**Tarefa mais favorável:** fazer.
**Tarefa mais difícil:** recomendar.
**Modelo mais estável em mediana:** Tucano, mas com outliers graves.
**Modelo com maiores oscilações:** Qwen.
**Cuidado principal:** interpretar PPL relativa, não PPL absoluta entre modelos.
