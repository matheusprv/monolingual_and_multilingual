# Experiments: HateBR and causal models

This directory compares causal models on HateBR for both classification and causal language modeling on the configured corpus.

## Files

- `pipeline.py`: entry point; reads the YAML, prepares data, and runs each stage for every model.
- `experiment.yaml`: configuration for models, data, training, generation, and causal evaluation.
- `experiment_config.py`: validates and converts the YAML into a typed configuration.
- `dataset.py`: loads, cleans, splits, and saves datasets.
- `model.py`: loads Hugging Face or native Candeia (xLSTM/Transformer) models.
- `hatebr_lora_causal_multimodel.py`: prompts, classification, LoRA fine-tuning, and metrics.
- `causal_sliding.py`: perplexity (PPL) and bits-per-byte (BPB) computation with sliding windows.
- `utils.py`: RAM/VRAM cleanup between runs.

## Pipeline

The pipeline sets the seed, prepares the data once, and iterates through `models`. Therefore, within one run, every model uses the same splits. Results, adapters, metrics, and predictions are saved under `output_dir`.

### HateBR dataset

The dataset is downloaded from Hugging Face, retaining `comentario` as `text` and `label_final` as `label`. Rows without text or labels, as well as empty texts, are removed. The data is split with `train_test_split`, stratified by `label`: first into train/test, then train/validation. The CSV files actually used are saved under `output_dir/splits`.

### Classification with a causal model

Each example becomes a prompt requesting either `ofensivo` or `não ofensivo`. During training, loss is masked over the prompt and computed only on answer tokens; LoRA is configured as `CAUSAL_LM`. Thus, no classification head is added: the model still predicts the next token and classifies by generating the textual label.

### Causal evaluation: PPL and BPB

For each text, *causal sliding* tokenizes it and processes windows of up to `max_length`, advancing by `stride` tokens. Loss is accumulated only for the new tokens in each window, avoiding repeated context. Mean NLL produces PPL (`exp(mean NLL)`); BPB normalizes total NLL by the text's UTF-8 bytes (`NLL / (ln(2) × bytes)`).

## YAML

`experiment.yaml` contains:

- `models`: model IDs/paths from Hugging Face, or detailed native Candeia entries;
- `output_dir` and `seed`: output directory and reproducibility;
- `dataset`: dataset name, test fraction, and validation fraction;
- `model`: 4-bit quantization setting;
- `generation`: prompt and generated-response limits;
- `training`: LoRA and training hyperparameters;
- `causal`: Parquet file, text column, scenarios, row limit, stride, and window size.

## Actions

- `causal`: measures PPL/BPB for the base model on the Parquet corpus.
- `predict`: evaluates zero-shot classification on the HateBR test set.
- `train`: trains and saves the LoRA adapter using the training and validation sets.
- `evaluate`: loads the adapter, evaluates classification on the test set, and measures PPL/BPB for the fine-tuned model.
- `all`: runs `predict`, `train`, and `evaluate`, in that order, for each model.

## Running

From the repository root, provide the YAML file and desired action:

```bash
python3 -m src.experiments.pipeline src/experiments/experiment.yaml causal
python3 -m src.experiments.pipeline src/experiments/experiment.yaml predict
python3 -m src.experiments.pipeline src/experiments/experiment.yaml train
python3 -m src.experiments.pipeline src/experiments/experiment.yaml evaluate
python3 -m src.experiments.pipeline src/experiments/experiment.yaml all
```

`evaluate` requires that `train` has already produced the corresponding adapter. Adjust relative paths in the YAML according to the directory from which the command is run.

### Checkpoints Candeia

Strings in `models` retain the original behavior and are loaded with Hugging Face.
For a checkpoint produced by Candeia, use a YAML map with its `model.pt` and
SentencePiece `tokenizer.model`:

```yaml
models:
  - name: Manaca
    backend: hf
    path: ../models/models/menezesbruno/manaca-1b-base
    tokenizer: ../models/models/menezesbruno/manaca-1b-base
  - name: Candeia-xLSTM-350M
    backend: xlstm
    path: /media/data/matheusvieira/candeia-xlstm-350M/model.pt
    tokenizer: /media/data/matheusvieira/candeia-xlstm-350M/tokenizer.model
  - name: Candeia-Transformer-350M
    backend: candeia_transformer
    path: /media/data/matheusvieira/candeia-transformer-350M/model.pt
    tokenizer: /media/data/matheusvieira/candeia-transformer-350M/tokenizer.model
```

Both native backends support `causal`, `predict`, `train`, `evaluate`, and
`all`. Native training performs full fine-tuning (not LoRA) and writes the
best complete state dictionary to
`output_dir/models/<name>/hatebr_full_finetuned.pt`; `evaluate` reloads that
file automatically. They require the package that exposes `xlstm_ptbr` to be
installed in the environment. If the context size is not exposed by a
particular Candeia release, set `causal.max_length` explicitly.
