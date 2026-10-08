import logging
import math
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from .model import Model
from .experiment_config import ModelSpec
from .utils import clean_memory


def eval_causal_sliding(
    model: Model,
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
        max_len = model.max_context_length
    if max_len is None:
        raise ValueError(
            "Não foi possível inferir o comprimento de contexto deste modelo. "
            "Defina causal.max_length no YAML."
        )

    if stride <= 0:
        raise ValueError("stride must be positive")

    if stride > max_len:
        raise ValueError("stride cannot be greater than max_len")

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

            # Isto pode gerar OOM se todo o texto tokenizado não couber na GPU.
            inputs = model.prepare_inputs(text)

            num_tokens = inputs["input_ids"].size(1)

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

                input_ids = inputs["input_ids"][
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
                    with torch.inference_mode():
                        window_attention_mask = inputs.get("attention_mask")
                        if window_attention_mask is not None:
                            window_attention_mask = window_attention_mask[
                                :, begin_window:end_window
                            ]
                        window_nll, window_loss_tokens = model.causal_nll(
                            input_ids,
                            target_ids,
                            attention_mask=window_attention_mask,
                        )

                    total_nll += window_nll
                    total_loss_tokens += window_loss_tokens

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
                "window_attention_mask",
            ):
                if variable_name in locals():
                    del locals()[variable_name]

            clean_memory()

    return results


def execute_experiment(
    model_spec: ModelSpec,
    quantization,
    corpus: pd.DataFrame,
    output_folder: Path = Path("./results"),
    scenarios: list | str = "all",
    stride: int = 512,
    max_len: int | None = None,
    output_file_name: str | None = None,
):

    # preparing the selected corpus to evaluate
    dataset = corpus.copy()
    if scenarios != "all":
        dataset = dataset[dataset["scenario"].isin(scenarios)]

    model = None
    try:
        print(f"⏳ Loading {model_spec.name}")
        model = Model(model_spec, quantization)

        print("🖥️ Executing experiment")
        results = eval_causal_sliding(model, dataset, stride=stride, max_len=max_len)

    except Exception:
        logging.exception(
            "Error model %s from %s with quantization=%s",
            model_spec.name,
            model_spec.path,
            quantization,
        )
        return None

    finally:
        del model
        clean_memory()

    print("💾 Saving model results")
    results_df = pd.DataFrame(results)
    results_df["model"] = model_spec.name
    output_folder.mkdir(parents=True, exist_ok=True)
    file_name = output_file_name or model_spec.name
    results_df.to_csv(output_folder / f"{file_name}.csv", index=False)
    return results_df
