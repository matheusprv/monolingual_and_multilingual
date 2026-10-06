import logging
import math
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from .model import Model
from .utils import clean_memory


def eval_causal_sliding(
    model: Model,
    corpus: pd.DataFrame,
    stride: int = 512,
    max_len: int | None = None,
):
    language_model = model.model
    tokenizer = model.tokenizer

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
            max_len = language_model.config.max_position_embeddings
        except AttributeError:
            try:
                max_len = language_model.config.text_config.max_position_embeddings
            except AttributeError:
                max_len = tokenizer.model_max_length

    if stride <= 0:
        raise ValueError("stride must be positive")

    if stride > max_len:
        raise ValueError("stride cannot be greater than max_len")

    # More reliable for models loaded with device_map="auto".
    try:
        input_device = language_model.get_input_embeddings().weight.device
    except Exception:
        input_device = next(language_model.parameters()).device

    model.set_evaluation_mode()

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
                        outputs = language_model(**model_kwargs)

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

            clean_memory()

    return results


def execute_experiment(
    model_name,
    model_url,
    quantization,
    corpus: pd.DataFrame,
    output_folder: Path = Path("./results"),
    scenarios: list | str = "all",
):

    # preparing the selected corpus to evaluate
    dataset = corpus.copy()
    if scenarios != "all":
        dataset = dataset[dataset["scenario"].isin(scenarios)]

    model = None
    try:
        print(f"⏳ Loading {model_name}")
        model = Model(model_url, quantization)

        print("🖥️ Executing experiment")
        results = eval_causal_sliding(model, dataset)

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
        clean_memory()

    print("💾 Saving model results")
    results_df = pd.DataFrame(results)
    results_df.to_csv( output_folder / f"{model_name}.csv", index=False)

    results_df["model"] = model_name
    return results_df
