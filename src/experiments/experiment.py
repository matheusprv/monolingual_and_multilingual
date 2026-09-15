# pip install -U transformers python-dotenv sentencepiece accelerate bitsandbytes

# Connecting to huggin face and downloading dataset
import os
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import login
from pathlib import Path


# Checking if it is running on Google Colab or locally
try:
    from google.colab import userdata
    import gdown

    hf_token = userdata.get('HF_TOKEN')

    url = 'https://drive.google.com/file/d/18f60tbAjVKYgfqu2ILngBw2wLjwF5lv-/view?usp=sharing'
    gdown.download(url, quiet=True, fuzzy=True)

    dataset = "./corpora.parquet"
    outputs_folder = Path("./")

except ModuleNotFoundError as e:
    hf_token = os.getenv("HF_TOKEN")
    #dataset = "../../corpora/GeneratedCorpus/corpora.parquet"
    dataset = "./corpora_v3.parquet"
    # outputs_folder = Path("../../results")
    outputs_folder = Path("./results4")


login(hf_token)
os.makedirs(outputs_folder, exist_ok=True)

import gc
import torch
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import json
import logging
from tqdm.auto import tqdm
from glob import glob
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# pip install bitsandbytes accelerate


logging.basicConfig(
    filename="experiment_errors.log",
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)

def clean_memory():
    """Force RAM and VRAM to clean"""
    gc.collect()                # Clean not used python objects
    torch.cuda.empty_cache()    # Clean Pytorch alocation cache
    print("Memory cleaned 🧹!")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Execution environment: {device.upper()}")


df = pd.read_parquet(dataset)

SCENARIOS = df["scenario"].unique().tolist()
print(SCENARIOS)

LANGAUGES = df["language"].unique().tolist()
print(LANGAUGES)

TASKS = df["task"].unique().tolist()
print(TASKS)

TASK_TYPES = df["task_type"].unique().tolist()
print(TASK_TYPES)

df.tail()


def load_model_and_tokenizer(model_url: str, quantize: bool = False):
    print(f"Starting accelerated download for: {model_url}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_url,
        use_fast=False, 
        trust_remote_code=True
    )

    bnb_config = None
    if quantize:
        print("Quantizing model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_url,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 
    )

    model.eval()
    print("Download and loading complete.")
    return model, tokenizer


def eval_causal_sliding(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    corpus: pd.DataFrame,
    stride: int = 512,
    max_len: int | None = None,
):
    results = {
        "text_id": [],
        "num_words": [],
        "num_bytes": [],
        "num_tokens": [],
        "tokens_per_word": [],
        "average_nll": [],
        "total_nll": [],
        "ppl": [],
        "bpb": [],
    }

    if max_len is None:
        try:
            max_len = model.config.max_position_embeddings
        except AttributeError:
            try:
                max_len = model.config.text_config.max_position_embeddings
            except AttributeError:
                max_len = tokenizer.model_max_length

    if stride <= 0:
        raise ValueError("stride must be positive")

    if stride > max_len:
        raise ValueError("stride cannot be greater than max_len")

    # More reliable for models loaded with device_map="auto".
    try:
        input_device = model.get_input_embeddings().weight.device
    except Exception:
        input_device = next(model.parameters()).device

    model.eval()

    progress_bar = tqdm(
        corpus.iterrows(),
        total=len(corpus),
        desc="Processing corpus",
        unit="texts",
    )

    for text_id, row in progress_bar:
        text = row["text"]

        # Always store the ID, even when the text fails.
        results["text_id"].append(text_id)

        try:
            num_bytes = len(text.encode("utf-8"))
            num_words = len(text.split())

            # This may itself raise an OOM if the full encoded text is moved
            # to the GPU.
            inputs = tokenizer(
                text,
                return_tensors="pt",
            ).to(input_device)

            num_tokens = inputs.input_ids.size(1)

            tokens_per_word = (
                num_tokens / num_words
                if num_words > 0
                else float("nan")
            )

            total_nll = 0.0
            total_loss_tokens = 0
            prev_end_window = 0

            for begin_window in range(0, num_tokens, stride):
                end_window = min(
                    begin_window + max_len,
                    num_tokens,
                )

                target_len = end_window - prev_end_window

                input_ids = inputs.input_ids[
                    :, begin_window:end_window
                ]

                target_ids = input_ids.clone()
                target_ids[:, :-target_len] = -100

                num_loss_tokens = (
                    target_ids[:, 1:] != -100
                ).sum().item()

                # Avoid calling the model when no next-token loss can
                # be calculated.
                if num_loss_tokens > 0:
                    model_kwargs = {
                        "input_ids": input_ids,
                        "labels": target_ids,
                    }

                    if "attention_mask" in inputs:
                        model_kwargs["attention_mask"] = (
                            inputs.attention_mask[
                                :, begin_window:end_window
                            ]
                        )

                    with torch.inference_mode():
                        outputs = model(**model_kwargs)

                    total_nll += (
                        outputs.loss.item() * num_loss_tokens
                    )
                    total_loss_tokens += num_loss_tokens

                    del outputs

                del input_ids, target_ids

                prev_end_window = end_window

                if end_window == num_tokens:
                    break

            if total_loss_tokens > 0:
                average_nll = total_nll / total_loss_tokens

                try:
                    ppl = math.exp(average_nll)
                except OverflowError:
                    ppl = float("inf")
            else:
                average_nll = float("nan")
                ppl = float("nan")

            if num_bytes > 0 and total_loss_tokens > 0:
                bpb = total_nll / (
                    math.log(2) * num_bytes
                )
            else:
                bpb = float("nan")

            results["num_words"].append(num_words)
            results["num_bytes"].append(num_bytes)
            results["num_tokens"].append(num_tokens)
            results["tokens_per_word"].append(tokens_per_word)
            results["average_nll"].append(average_nll)
            results["total_nll"].append(total_nll)
            results["ppl"].append(ppl)
            results["bpb"].append(bpb)

            del inputs

        except torch.cuda.OutOfMemoryError:
            progress_bar.write(
                f"CUDA OOM for text_id={text_id!r}; "
                "recording NaN and continuing."
            )

            # Every result list must receive one value to remain aligned.
            results["num_words"].append(float("nan"))
            results["num_bytes"].append(float("nan"))
            results["num_tokens"].append(float("nan"))
            results["tokens_per_word"].append(float("nan"))
            results["average_nll"].append(float("nan"))
            results["total_nll"].append(float("nan"))
            results["ppl"].append(float("nan"))
            results["bpb"].append(float("nan"))

        finally:
            # Remove any tensors left alive after either success or failure.
            for variable_name in (
                "inputs",
                "input_ids",
                "target_ids",
                "outputs",
                "model_kwargs",
            ):
                if variable_name in locals():
                    del locals()[variable_name]

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def execute_experiment(model_name, model_url, quantization, corpus:pd.DataFrame, output_folder:Path=outputs_folder, scenarios:list|str="all"):

    # preparing the selected corpus to evaluate
    dataset = corpus.copy()
    if scenarios != "all":
        dataset = dataset[dataset["scenario"].isin(scenarios)]

    model, tokenizer = None, None
    try:
        print(f"⏳ Loading {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_url, quantization)

        print("🖥️ Executing experiment")
        results = eval_causal_sliding(model, tokenizer, dataset)

    except Exception:
        logging.exception(
            "Error model %s from %s with quantization=%s",
            model_name,
            model_url,
            quantization,
        )
        return None

    finally:
        del model
        del tokenizer
        clean_memory()

    print("💾 Saving model results")
    results_df = pd.DataFrame(results)
    results_df.to_csv( output_folder / f"{model_name}.csv", index=False)

    results_df["model"] = model_name
    return results_df


models = {   
    "Cabrita": ("../../../models/models/22h/open-cabrita3b", False),
    "Gemma4-12B": ("../../../models/models/google/gemma-4-12B", False),
    "Gemma4-12B-it": ("../../../models/models/google/gemma-4-12B-it", False),
    "Sabia 7B": ("../../../models/models/maritaca-ai/sabia-7b", False),
    "Manaca": ("../../../models/models/menezesbruno/manaca-1b-base", False),
    "Qwen3.5-9B-it": ("../../../models/models/Qwen/Qwen3.5-9B", False),
    "Qwen3.5-9B": ("../../../models/models/Qwen/Qwen3.5-9B-Base", False),
    "Tucano-2b4": ("../../../models/models/TucanoBR/Tucano-2b4", False),
    "Tucano-2b4-it": ("../../../models/models/TucanoBR/Tucano-2b4-Instruct", False)
}




#execute_experiment("Cabrita", *(models["Cabrita"]), df)
execute_experiment("Gemma4-12B", *(models["Gemma4-12B"]), df)        #- Nao feito
execute_experiment("Gemma4-12B-it", *(models["Gemma4-12B-it"]), df)  #- Nao feito
# execute_experiment("Sabia 7B", *(models["Sabia 7B"]), df)
# execute_experiment("Manaca", *(models["Manaca"]), df)
# execute_experiment("Qwen3.5-9B-it", *(models["Qwen3.5-9B-it"]), df)
# execute_experiment("Qwen3.5-9B", *(models["Qwen3.5-9B"]), df)
# execute_experiment("Tucano-2b4", *(models["Tucano-2b4"]), df)
# execute_experiment("Tucano-2b4-it", *(models["Tucano-2b4-it"]), df)
