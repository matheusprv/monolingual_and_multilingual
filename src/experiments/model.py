"""Backends de modelos usados pelos experimentos.

Os experimentos trabalham com uma pequena interface comum, em vez de depender
diretamente de ``transformers``. Isso permite usar checkpoints Candeia
(xLSTM e Transformer) distribuídos como ``model.pt``.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .experiment_config import ModelSpec


class Model:
    def __init__(
        self,
        spec: ModelSpec,
        quantization: bool,
        *,
        native_training: bool = False,
        native_weights_path: Path | None = None,
    ):
        self.spec = spec
        self.backend = spec.backend
        self._is_model_evaluating = False
        self._native_device: torch.device | None = None
        self._native_max_context_length: int | None = None
        self._native_training = native_training

        if self.backend == "hf":
            self.tokenizer = self.load_hf_tokenizer(spec.path)
            self.model = self.load_hf_model(spec.path, quantization)
        elif self.backend in {"xlstm", "candeia_transformer"}:
            if quantization:
                print("Aviso: use_4bit é ignorado para checkpoints Candeia nativos.")
            self.tokenizer, self.model = self.load_candeia(spec)
            if native_weights_path is not None:
                self.load_native_weights(native_weights_path)
        else:
            raise ValueError(f"Backend não suportado: {self.backend!r}")
        self.set_evaluation_mode()

    @property
    def supports_lora(self) -> bool:
        """PEFT/Trainer só é suportado pelo backend Transformers neste projeto."""
        return self.backend == "hf"

    def load_hf_tokenizer(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def load_hf_model(self, model_path: str, quantization: bool):
        bnb_config = None
        if quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    def load_candeia(self, spec: ModelSpec):
        """Carrega o formato nativo salvo pelos treinamentos Candeia."""
        if not spec.tokenizer:
            raise ValueError(f"O modelo Candeia {spec.name!r} requer 'tokenizer' no YAML.")
        try:
            import sentencepiece as spm
        except ImportError as error:
            raise ImportError("Instale sentencepiece para usar os modelos Candeia.") from error

        checkpoint = torch.load(spec.path, map_location="cpu", weights_only=False)
        try:
            model_config = checkpoint["config"]["model"]
            state_dict = checkpoint["model"]
        except KeyError as error:
            raise ValueError(
                f"Checkpoint Candeia inválido em {spec.path!r}; esperava as chaves "
                "'config.model' e 'model'."
            ) from error

        if self.backend == "xlstm":
            try:
                from xlstm_ptbr.models.xlstm.lm import XlstmLmConfig, XlstmLmHeadModel
            except ImportError as error:
                raise ImportError(
                    "Não foi possível importar xlstm_ptbr. Instale o repositório que "
                    "fornece as classes do checkpoint Candeia."
                ) from error
            mode = "train" if self._native_training else "inference"
            config = XlstmLmConfig(**{**model_config, "mode": mode})
            native_model = XlstmLmHeadModel(config)
        else:
            try:
                from xlstm_ptbr.models.transformer.lm import (
                    TransformerLmConfig,
                    TransformerLmHeadModel,
                )
            except ImportError as error:
                raise ImportError(
                    "Não foi possível importar xlstm_ptbr. Instale o repositório que "
                    "fornece as classes do checkpoint Candeia."
                ) from error
            config = TransformerLmConfig(
                **{**model_config, "activation_checkpointing": False}
            )
            native_model = TransformerLmHeadModel(config)

        native_model.load_state_dict(state_dict)
        for attribute in ("max_position_embeddings", "context_length", "max_sequence_length", "block_size"):
            value = getattr(config, attribute, None)
            if isinstance(value, int) and value > 0:
                self._native_max_context_length = value
                break
        self._native_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        native_model.to(self._native_device)
        # O exemplo do Candeia Transformer carrega os pesos já em bf16.
        if self.backend == "candeia_transformer" and self._native_device.type == "cuda":
            native_model.to(dtype=torch.bfloat16)
        return spm.SentencePieceProcessor(model_file=spec.tokenizer), native_model

    def load_native_weights(self, path: Path) -> None:
        """Restaura pesos completos produzidos pelo ajuste fino nativo."""
        if self.backend == "hf":
            raise ValueError("Pesos nativos não podem ser carregados em um backend HF.")
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint treinado não encontrado em {path}.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)

    def save_native_weights(self, path: Path) -> None:
        """Salva somente os pesos, mantendo a configuração no checkpoint-base YAML."""
        if self.backend == "hf":
            raise ValueError("Use save_pretrained para o backend HF.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": self.model.state_dict(), "backend": self.backend},
            path,
        )

    def set_evaluation_mode(self):
        if not self._is_model_evaluating:
            self.model.eval()
            self._is_model_evaluating = True

    def set_train_mode(self):
        if self._is_model_evaluating:
            self.model.train()
            self._is_model_evaluating = False

    @property
    def pad_token_id(self) -> int:
        if self.backend == "hf":
            return int(self.tokenizer.pad_token_id)
        pad_id = self.tokenizer.pad_id()
        return int(pad_id if pad_id >= 0 else self.eos_token_id)

    @property
    def eos_token_id(self) -> int:
        if self.backend == "hf":
            return int(self.tokenizer.eos_token_id)
        eos_id = self.tokenizer.eos_id()
        if eos_id < 0:
            raise ValueError("O tokenizer SentencePiece não define um token EOS.")
        return int(eos_id)

    @property
    def input_device(self) -> torch.device:
        if self.backend != "hf":
            assert self._native_device is not None
            return self._native_device
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return next(self.model.parameters()).device

    @property
    def max_context_length(self) -> int | None:
        """Devolve o limite conhecido; ``None`` exige causal.max_length no YAML."""
        if self._native_max_context_length is not None:
            return self._native_max_context_length
        config = getattr(self.model, "config", None)
        for attribute in ("max_position_embeddings", "context_length", "max_sequence_length", "block_size"):
            value = getattr(config, attribute, None)
            if isinstance(value, int) and value > 0:
                return value
        text_config = getattr(config, "text_config", None)
        value = getattr(text_config, "max_position_embeddings", None)
        if isinstance(value, int) and value > 0:
            return value
        if self.backend == "hf":
            value = getattr(self.tokenizer, "model_max_length", None)
            if isinstance(value, int) and value < 1_000_000:
                return value
        return None

    def prepare_inputs(
        self, text: str, *, truncation: bool = False, max_length: int | None = None
    ) -> dict[str, torch.Tensor]:
        if self.backend == "hf":
            encoded = self.tokenizer(
                text, return_tensors="pt", truncation=truncation, max_length=max_length
            ).to(self.input_device)
            return dict(encoded)

        ids = self.tokenizer.encode(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.input_device)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def encode_ids(self, text: str) -> list[int]:
        """Tokeniza sem mover tensores à GPU; usado pelo treino nativo."""
        if self.backend == "hf":
            return list(self.tokenizer(text, add_special_tokens=False).input_ids)
        return list(self.tokenizer.encode(text))

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        if self.backend == "hf":
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return self.tokenizer.decode(token_ids)

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        max_new_tokens: int,
    ) -> torch.Tensor:
        if self.backend == "hf":
            kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": self.pad_token_id,
                "eos_token_id": self.eos_token_id,
            }
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            return self.model.generate(**kwargs)

        with self._native_autocast():
            output = self.model.generate(
                input_ids, max_new_tokens=max_new_tokens, temperature=0.0
            )
        sequence = output[0] if isinstance(output, tuple) else output
        return sequence.unsqueeze(0) if sequence.ndim == 1 else sequence

    def causal_nll(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[float, int]:
        """Retorna NLL total e número de predições válidas para uma janela."""
        num_loss_tokens = int((labels[:, 1:] != -100).sum().item())
        if num_loss_tokens == 0:
            return 0.0, 0

        if self.backend == "hf":
            kwargs: dict[str, Any] = {"input_ids": input_ids, "labels": labels}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            outputs = self.model(**kwargs)
            return float(outputs.loss.item() * num_loss_tokens), num_loss_tokens

        with self._native_autocast():
            logits = self._native_forward_logits(input_ids)
            loss = F.cross_entropy(
                logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        return float(loss.item()), num_loss_tokens

    def native_causal_loss(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Loss média diferenciável usada pelo ajuste fino completo Candeia."""
        if self.backend == "hf":
            raise ValueError("native_causal_loss é exclusivo dos backends Candeia.")
        with self._native_autocast():
            logits = self._native_forward_logits(input_ids)
            return F.cross_entropy(
                logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )

    def _native_forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Executa o forward nativo e normaliza os formatos de saída Candeia.

        O kernel de treino do xLSTM processa blocos de 16 tokens. Padding é
        aplicado apenas à direita e os logits correspondentes são descartados;
        como o modelo é causal, isso não altera as predições dos tokens reais.
        """
        original_length = input_ids.size(1)
        model_input_ids = input_ids
        if self.backend == "xlstm":
            extra_tokens = (-original_length) % 16
            if extra_tokens:
                padding = torch.full(
                    (input_ids.size(0), extra_tokens),
                    self.pad_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                model_input_ids = torch.cat((input_ids, padding), dim=1)

        outputs = self.model(model_input_ids)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, Mapping):
            if "logits" not in outputs:
                raise ValueError(
                    "O forward Candeia retornou um mapa sem a chave 'logits': "
                    f"{list(outputs.keys())}."
                )
            logits = outputs["logits"]
        elif isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        else:
            logits = outputs
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                "Não foi possível obter um tensor de logits do forward Candeia; "
                f"recebi {type(logits).__name__}."
            )
        return logits[:, :original_length, :]

    def _native_autocast(self):
        if self._native_device is not None and self._native_device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()


class LoRA_Model(Model):
    def __init__(self, spec: ModelSpec, quantization: bool, adapter_path: str):
        super().__init__(spec, quantization)
        if not self.supports_lora:
            raise ValueError(
                f"LoRA/PEFT ainda não é suportado para o backend {self.backend!r}. "
                "Use as ações 'causal' ou 'predict', ou implemente treino nativo."
            )
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self._is_model_evaluating = False
        self.set_evaluation_mode()
