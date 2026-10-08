import torch, sentencepiece as spm
from xlstm_ptbr.models.xlstm.lm import XlstmLmConfig, XlstmLmHeadModel

MODEL_PATH = "/media/data/matheusvieira/candeia-xlstm-350M/model.pt"
TOKENIZER_PATH = "/media/data/matheusvieira/candeia-xlstm-350M/tokenizer.model"

print("⏳ Carregando modelo")

ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
cfg = XlstmLmConfig(**{**ck["config"]["model"], "mode": "inference"})  # generate() exige modo de inferência
model = XlstmLmHeadModel(cfg)
model.load_state_dict(ck["model"])
model.eval().to("cuda")

print("⏳ Carregando tokenizador")
sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
ids = torch.tensor([sp.encode("a capital do brasil é")], device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    out, _ = model.generate(ids, max_new_tokens=20, temperature=0.0)  # 0.0 = guloso
print(sp.decode(out[0].tolist()))



###################

import torch, sentencepiece as spm
from xlstm_ptbr.models.transformer.lm import TransformerLmConfig, TransformerLmHeadModel

MODEL_PATH = "/media/data/matheusvieira/candeia-transformer-350M/model.pt"
TOKENIZER_PATH = "/media/data/matheusvieira/candeia-transformer-350M/tokenizer.model"

ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
cfg = TransformerLmConfig(**{**ck["config"]["model"], "activation_checkpointing": False})
model = TransformerLmHeadModel(cfg)
model.load_state_dict(ck["model"])
model.to("cuda", dtype=torch.bfloat16).eval()

sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
ids = torch.tensor([sp.encode("a capital do brasil é")], device="cuda")
out, _ = model.generate(ids, max_new_tokens=20, temperature=0.0)  # 0.0 = guloso
print(sp.decode(out[0].tolist()))