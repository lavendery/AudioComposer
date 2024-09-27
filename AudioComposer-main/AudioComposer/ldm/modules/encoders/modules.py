import os.path

import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel



class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class FrozenFLANEmbedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, weights_path, version="google/flan-t5-large", device="cuda", max_length=512, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(weights_path + version)
        self.transformer = T5EncoderModel.from_pretrained(weights_path + version)

        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True, 
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        
        attention_mask = batch_encoding.attention_mask.to(self.device)
        outputs = self.transformer(input_ids=tokens, attention_mask=attention_mask) 

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)
