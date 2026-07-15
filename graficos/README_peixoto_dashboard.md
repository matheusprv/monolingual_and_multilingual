# Peixoto Results Dashboard

Interactive Python dashboard for comparing causal language model results across:

- Grammar swap
- Lexical swap
- Translation

The dashboard supports:

- Raw metrics and percentage comparison against **PB NATIVO**
- Model, language, task, and scenario filters
- Mean or median aggregation
- Standard deviation display
  - error bars in bar and line charts
  - hover information and optional std heatmap in heatmaps
  - full distribution in box plots
- Optional log scale
- Bar charts, grouped bars, heatmaps, box plots, line/profile charts, scatter plots, and ranking tables
- Export of filtered data, aggregated data, HTML charts, and PDF charts when `kaleido` is installed

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run results_dashboard.py
```

## Input files

Place these CSVs in the same folder as the script:

```text
grammar_swap_causalLM_results_compiled.csv
lexical_swap_causalLM_results_compiled.csv
translation_causalLM_results_compiled.csv
```

The app also supports manual CSV upload from the sidebar.

## PB NATIVO comparison

For each metric, the comparison is calculated as:

```text
(row_value / pb_nativo_value - 1) * 100
```

The PB baseline is matched by:

```text
Scenario + Model + Task
```

So each language is compared to PB NATIVO under the same experimental condition.
