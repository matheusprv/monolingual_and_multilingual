import warnings

warnings.filterwarnings(
    "ignore",
    message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

import logging

logging.getLogger(
    "bitsandbytes.autograd._functions"
).setLevel(logging.ERROR)

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig,
)
import torch



class Model:
    def __init__(self, model_name, device_map="auto"):




        self.processor = AutoProcessor.from_pretrained(model_name)

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map=device_map,
            quantization_config=quantization_config,
        )

        print(self.model.get_memory_footprint() / 1024**3, "GB")

    def message(self, text):
        messages = [
            {"role": "user", "content": text},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256
            )

        response = self.processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=False
        )

        response = self.processor.parse_response(
            response,
            prefix=inputs["input_ids"]
        )

        return response["content"]