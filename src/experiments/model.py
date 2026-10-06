from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel

class Model:
    def __init__(self, model_url:str, quantization:bool):
        self.tokenizer = self.load_tokenizer(model_url)
        self.model = self.load_model(model_url, quantization)
        self._is_model_evaluating = False
        self.set_evaluation_mode()
    
    def load_tokenizer(self, model_url):
        tokenizer = AutoTokenizer.from_pretrained(
                model_url,
                use_fast=False, 
                trust_remote_code=True
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        return tokenizer
    
    
    def load_model(self, model_url, quantization):
        bnb_config = None
        if quantization:
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
    
        return model
    
    
    def set_evaluation_mode(self):
        if not self._is_model_evaluating:
            self.model.eval()
            self._is_model_evaluating = True
    
    def set_train_mode(self):
        if self._is_model_evaluating:
            self.model.train() 
            self._is_model_evaluating = False
    

class LoRA_Model(Model):
    def __init__(self, model_url:str, quantization:bool, adapter_path:str):
        super().__init__(model_url, quantization)

        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
        )

        self._is_model_evaluating = False
        self.set_evaluation_mode()
