import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence


import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict


import random
import numpy as np

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    HfArgumentParser, 
    TrainingArguments, 
    pipeline
)

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class ZeroOutputLinear(nn.Linear):
    def forward(self, input):
        output = super().forward(input)
        return torch.zeros_like(output)
    
class ToC_qwen(nn.Module):    
    def __init__(self,
                 model_id: str = 'Qwen/Qwen2-7B-Instruct',
                 cache_dir: str = 'Qwen2-7B-Instruct/original',
                 cue_types: list = ["linguistic", "contextual", "emotional"],
                 max_cue_len: int = 100,
                ):
        super().__init__()

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = cache_dir,token = None)
        self.qwen = AutoModelForCausalLM.from_pretrained(model_id, cache_dir = cache_dir, token = None)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.cue_types = cue_types

        for param in self.qwen.model.parameters():
            param.requires_grad = False

        
        #zero_output_linear = ZeroOutputLinear(self.qwen.config.hidden_size, 2**4-1)
        #for param in zero_output_linear.parameters():
        #    param.requires_grad = False
        #zero_output_linear2 = ZeroOutputLinear(self.qwen.config.hidden_size, 14-1)
        #for param in zero_output_linear2.parameters():
        #    param.requires_grad = False
        

        self.dim_reduction_nets = nn.ModuleList([
                                        nn.Linear(self.qwen.config.hidden_size, 2**4-1, bias=True),
                                        nn.Linear(self.qwen.config.hidden_size, 2**4-1, bias=True),
                                        nn.Linear(self.qwen.config.hidden_size, 14-1, bias=True)
                                        
                                        ])
        
        
    def forward(self, 
                cue_ids: torch.Tensor, 
                cue_masks: torch.Tensor, 
                prompt_ids: torch.Tensor, 
                prompt_masks: torch.Tensor,
                # max_cue_len: int = 64,
                ):
        #print('get the 3 cue indices')
       
        
        concatenated_input, input_mask = self.tensor_of_cues(prompt_ids,prompt_masks, cue_ids, cue_masks)

    

        result = self.qwen.forward(inputs_embeds=concatenated_input,
                                    attention_mask=input_mask,
                                    return_dict = True)


        logits = result['logits']

        indices = torch.arange(input_mask.size(1), device=input_mask.device)

        masked_indices = input_mask * indices
        max_indices = masked_indices.argmax(dim=1)
        logits = logits[torch.arange(max_indices.size(0)), max_indices]


        pred = torch.argmax(logits, dim=-1)
        
        return logits, pred
    

    
    def tensor_of_cues(self, prompt_ids, prompt_masks, output_cue_ids, cue_masks):

        binary_mask = cue_masks.sum(dim=1) > 0

        output_cue_embeddings = self.qwen.model.embed_tokens(output_cue_ids)

        output_cue_embeddings = output_cue_embeddings.transpose(0,1)
        
        cue_tensors = [F.pad(net(cue_embed),(0,1), value=1) for net, cue_embed in zip(self.dim_reduction_nets, output_cue_embeddings)]


        start_char = 'i'
        input_elements = []
        output_expr = 'bi'
        current_char = start_char
        for _ in range(len(cue_tensors)):
            next_char = chr(ord(current_char) + 1)
            input_elements.append(f'b{start_char}{next_char}')
            output_expr = output_expr + next_char
            current_char = next_char

        input_expr = ','.join(input_elements)
        einsum_expr = f'{input_expr}->{output_expr}'

        cue_tensor = torch.einsum(einsum_expr, *cue_tensors).flatten(start_dim=-len(cue_tensors), end_dim=-1)
        #cue_tensor = self.scale(cue_tensor)

        prompt_embeddings = self.qwen.model.embed_tokens(prompt_ids)

        concatenated_input = torch.cat([cue_tensor, prompt_embeddings],dim=1)
        input_mask = torch.cat([binary_mask,prompt_masks], dim = 1)
        return concatenated_input, input_mask
    
