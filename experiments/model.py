import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class Model:
    def __init__(self, model_path, tokenizer_path=None, quantization=False):
        self.tokenizer = self.load_tokenizer(tokenizer_path or model_path)
        self.model = self.load_model(model_path, quantization)
        self.max_context_length = self.get_max_context_length()
        self._is_model_evaluating = False
        self.set_evaluation_mode()

    def load_tokenizer(self, tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def load_model(self, model_path, quantization):
        quantization_config = None
        if quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        return AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.float16,
        )

    def get_max_context_length(self):
        configs = [self.model.config]
        text_config = getattr(self.model.config, "text_config", None)
        if text_config is not None:
            configs.append(text_config)

        for config in configs:
            for attribute in (
                "max_position_embeddings",
                "n_positions",
                "max_sequence_length",
                "seq_length",
            ):
                value = getattr(config, attribute, None)
                if isinstance(value, int) and 0 < value < 1_000_000:
                    return value

        value = self.tokenizer.model_max_length
        if isinstance(value, int) and 0 < value < 1_000_000:
            return value
        return None

    def prepare_inputs(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        device = next(self.model.parameters()).device
        return {name: value.to(device) for name, value in inputs.items()}

    def causal_nll(self, input_ids, target_ids, attention_mask=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids,
        )
        num_loss_tokens = (target_ids[:, 1:] != -100).sum().item()
        return output.loss.item() * num_loss_tokens, num_loss_tokens

    def set_evaluation_mode(self):
        if not self._is_model_evaluating:
            self.model.eval()
            self._is_model_evaluating = True

    def set_train_mode(self):
        if self._is_model_evaluating:
            self.model.train()
            self._is_model_evaluating = False


class LoRAModel(Model):
    def __init__(
        self,
        adapter_path,
        tokenizer_path=None,
        quantization=False,
        base_model_path=None,
    ):
        if base_model_path is None:
            base_model_path = PeftConfig.from_pretrained(
                adapter_path
            ).base_model_name_or_path

        super().__init__(base_model_path, tokenizer_path, quantization)
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.max_context_length = self.get_max_context_length()
        self._is_model_evaluating = False
        self.set_evaluation_mode()
