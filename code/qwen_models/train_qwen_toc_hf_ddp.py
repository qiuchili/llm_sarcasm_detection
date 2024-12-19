import torch
import torch.distributed
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from typing import List, Optional, Tuple, TypedDict

import argparse
import pandas as pd
import numpy as np
import json
import wandb
import os
import sys

from sklearn import metrics
from tqdm import tqdm

# from qwen.tokenizer import Tokenizer
from toc_qwen_hf import ToC_Qwen

class TextDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_seq_len, max_cue_len, yes_id,no_id):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(csv_file, encoding_errors='ignore')
        self.data.dropna(inplace=True)
        self.yes_id = yes_id
        self.no_id = no_id
        self.max_seq_len = max_seq_len
        self.max_cue_len = max_cue_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_row = self.data.iloc[idx]
        all_cues, main_prompt = prepare_prompts(data_row)
        encoded_cues = self.tokenizer(all_cues, max_length=self.max_cue_len, padding='max_length', truncation=True, return_tensors="pt")
        encoded_prompts = self.tokenizer.apply_chat_template([{'role':'user','content':main_prompt}], add_generation_prompt=True, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_seq_len)
        encoded_prompt_mask = encoded_prompts != self.tokenizer.pad_token_id
        
        # encoded_prompts = self.tokenizer(main_prompt, max_length=self.max_seq_len, padding='max_length', truncation=True, return_tensors="pt")
        
        label = data_row['Label']
        label = self.yes_id if label == 1 else self.no_id
        
        return encoded_cues['input_ids'], encoded_cues['attention_mask'], encoded_prompts[0], encoded_prompt_mask[0], label


def prepare_prompts(data_row, cue_types = ["linguistic", "contextual", "emotional"]):
    all_cues = []
    for cue_type in cue_types:
        cue_text = data_row[f'{cue_type}_cue'].replace(data_row[f'{cue_type}_cue_prompt'],'')
        all_cues.append(cue_text)

    main_prompt = f'''
                ###
                
                The current cue is shown above. Consider the information provided in the current cue above. Classify whether the input text is sarcastic or not.

                If you think the Input text is sarcastic, answer: yes 
                
                If you think the Input text is not sarcastic, answer: no 

                ###  
                Input:
                {data_row['Text']}.
 
                '''
    return all_cues, main_prompt

class Trainer():
    def __init__(self, 
                 task_name: str,
                 model_id: str = 'Qwen/Qwen2-7B-Instruct',
                 cache_dir: str = 'qwen/original',
                 batch_size: int = 16,
                 max_seq_len: int = 256,
                 max_cue_len: int = 32,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01,
                 num_epoch: int = 10,
                 output_path: str = 'output.csv',
                 pred_path: str = 'pred.csv',
                 *args, 
                 **kwargs) -> None:

        qwen_model = ToC_Qwen(model_id=model_id,cache_dir = cache_dir)
        self.yes_id = qwen_model.tokenizer.encode('yes')[-1]
        self.no_id  = qwen_model.tokenizer.encode('no')[-1]
        for param in qwen_model.qwen.model.parameters():
            param.requires_grad = False
        
        rank = torch.distributed.get_rank()
        self.device = torch.device('cuda')
        #device = torch.device('cpu')
        qwen_model.to(self.device)
        self.qwen_model = DDP(qwen_model, device_ids=[rank], output_device=rank)

        train_file_name = 'datasets/train_' + task_name + '_with_toc_cues.csv'
        test_file_name = 'datasets/test_' + task_name + '_with_toc_cues.csv'
        self.tokenizer = qwen_model.tokenizer
        self.tokenizer.padding_side = 'right'
        self.output_path = output_path
        self.pred_path = pred_path

        train_dataset = TextDataset(train_file_name,self.tokenizer, max_seq_len, max_cue_len, self.yes_id, self.no_id)
        
        self.train_sampler = DistributedSampler(train_dataset, shuffle=True)
        #self.train_sampler = RandomSampler(train_dataset)
        #self.train_sampler = None
        self.train_dataloader = DataLoader(train_dataset, 
                                           batch_size=batch_size, 
                                           num_workers=1, 
                                           pin_memory=True,
                                           shuffle=(self.train_sampler is None),
                                           sampler=self.train_sampler)
        test_dataset = TextDataset(test_file_name,self.tokenizer, max_seq_len, max_cue_len, self.yes_id, self.no_id)
        
        self.test_sampler = None
        self.test_dataloader = DataLoader(test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=(self.test_sampler is None), 
                                          sampler=self.test_sampler)

        self.loss_fct = nn.CrossEntropyLoss()
        trainable_params = [param for param in self.qwen_model.parameters() if param.requires_grad]
        self.optimizer = AdamW(params = trainable_params,
                            lr = learning_rate, 
                            weight_decay = weight_decay)

        self.num_epoch = num_epoch

    def train(self):
        
        for epoch in range(self.num_epoch):
            total_train_loss = self.train_step(
                                        epoch = epoch, 
                                        num_epochs = self.num_epoch)
            
            total_test_loss = self.test_step(
                                        epoch = epoch, 
                                        num_epochs = self.num_epoch)
            
            print(f"==============================Epoch {epoch+1}================================")
            print(f'''
                  ### Epoch {epoch+1}:
                  The final train loss is {total_train_loss}
                  The final Test loss is {total_test_loss}''')
            print("==============================================================================")
            
        
    def forward_step(self, batch):
        cue_ids, cue_masks, prompt_ids, prompt_masks, labels = batch
        cue_ids, cue_masks, prompt_ids, prompt_masks, labels = cue_ids.to(self.device), cue_masks.to(self.device), prompt_ids.to(self.device), prompt_masks.to(self.device), labels.to(self.device)
        logits, pred = self.qwen_model.forward(
                                                cue_ids = cue_ids, 
                                                cue_masks = cue_masks, 
                                                prompt_ids = prompt_ids, 
                                                prompt_masks = prompt_masks)
        logits = logits.view(logits.size(0), -1)
        loss = self.loss_fct(logits, labels)

        return pred, loss
        
    def train_step(self, epoch, num_epochs):
        self.train_dataloader.sampler.set_epoch(epoch)
        train_loss_sum = []
        progress_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
        for train_batch in progress_bar:
            _, train_loss = self.forward_step(train_batch)
            self.optimizer.zero_grad()  
            train_loss.backward()       
            self.optimizer.step()  

            train_loss = train_loss.detach().item()
            progress_bar.set_postfix(train_loss=train_loss)

            if torch.distributed.get_rank() == 0:
                wandb.log({'epoch': epoch+1, 'train_loss': train_loss})

            train_loss_sum.append(train_loss)
            torch.cuda.empty_cache()
        return np.mean(train_loss_sum)

    def test_step(self, epoch, num_epochs):
        test_loss_sum = []
        pred_list = []
        label_list = []
        progress_bar = tqdm(self.test_dataloader, desc=f"Test Epoch {epoch+1}/{num_epochs}")
        texts = []
        for test_batch in progress_bar:
            _, _, encoded_prompt, _, test_labels = test_batch

            prompts = self.tokenizer.batch_decode(encoded_prompt)
            prompts = [p.split('Input:\n')[1].split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[0] for p in prompts]
            texts.extend(prompts)
            label_list.extend(test_labels.tolist())
            pred, test_loss = self.forward_step(test_batch)
            test_loss = test_loss.detach().item()
            test_loss_sum.append(test_loss) 
            pred_list.extend([p.item() for p in pred])
        

        label_list = [1 if l == self.yes_id else 0 for l in label_list]
        pred_list = [1 if l == self.yes_id else 0 for l in pred_list]

        df = pd.DataFrame({'Text':texts,'Label':label_list,'pred':pred_list})
        df.to_csv(self.output_path.replace('.csv',f'_epoch_{epoch}.csv'))
        print(f'Evaluate the performance for epoch {epoch+1}!')
        self.eval_performance(label_list, pred_list, metric_path=self.pred_path.replace('.json',f'_epoch_{epoch}.json'))

        test_loss_avg = np.mean(test_loss_sum)

        if torch.distributed.get_rank() == 0:
            wandb.log({'test_loss': test_loss_avg})
        torch.cuda.empty_cache()
        return test_loss_avg

    def eval_performance(self, y_true, y_pred, metric_path=None):
        # Precision
        metric_dict = {}
        precision = metrics.precision_score(y_true, y_pred)
        print("Precision:\n\t", precision)
        metric_dict['Precision'] = precision

        # Recall
        recall = metrics.recall_score(y_true, y_pred)
        print("Recall:\n\t",  recall)
        metric_dict['Recall'] = recall

        # Accuracy
        accuracy = metrics.accuracy_score(y_true, y_pred)
        print("Accuracy:\n\t", accuracy)
        metric_dict['Accuracy'] = accuracy

        print("-------------------F1, Micro-F1, Macro-F1, Weighted-F1..-------------------------")
        print("-------------------**********************************-------------------------")

        # F1 Score
        f1 = metrics.f1_score(y_true, y_pred)
        print("F1 Score:\n\t", f1)
        metric_dict['F1'] = f1

        # Micro-F1 Score
        micro_f1 =  metrics.f1_score(y_true, y_pred, average='micro')
        print("Micro-F1 Score:\n\t",micro_f1)
        metric_dict['Micro-F1'] = micro_f1

        # Macro-F1 Score
        macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')
        print("Macro-F1 Score:\n\t", macro_f1)
        metric_dict['Macro-F1'] = macro_f1

        # Weighted-F1 Score
        weighted_f1 = metrics.f1_score(y_true, y_pred, average='weighted')
        print("Weighted-F1 Score:\n\t", weighted_f1)
        metric_dict['Weighted-F1'] = weighted_f1

        print("------------------**********************************-------------------------")
        print("-------------------**********************************-------------------------")

        # ROC AUC Score
        roc_auc = metrics.roc_auc_score(y_true, y_pred)
        print("ROC AUC:\n\t", roc_auc) 
        metric_dict['ROC-AUC'] = roc_auc

        # Confusion matrix
        print("Confusion Matrix:\n\t", metrics.confusion_matrix(y_true, y_pred))  
        
        if torch.distributed.get_rank() == 0:
            wandb.log({'Precision': precision,
                        'Recall': recall,
                        'Accuracy': accuracy,
                        'F1:': f1,
                        'Micro-F1': micro_f1,
                        'Macro-F1': macro_f1,
                        'Weighted-F1': weighted_f1,
                        'ROC-AUC': roc_auc})


        if metric_path is not None:
            json.dump(metric_dict,open(metric_path,'w'),indent=4)


def set_up():
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print('world_size:',world_size)
    print('local_rank:',local_rank)
    torch.distributed.init_process_group("nccl", rank = local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running toc based on qwen for sarcasm detection.')
    parser.add_argument('--task_name', metavar='T', type=str, help='task name', default='semeval')
    parser.add_argument('--model_name', metavar='M', type=str, help='model id', default='Qwen/Qwen2-7B-Instruct')
    parser.add_argument('--output_path', metavar='O', type=str, help='predictions path', default='output/output_toc_semeval.csv')
    parser.add_argument('--metric_path', metavar='P', type=str, help='metrics path', default='output/metric_toc_semeval.json')
    parser.add_argument('--cache_dir', metavar='C', type=str, help='cache dir', default='qwen/original')
    parser.add_argument('--run_name', metavar='N', type=str, help='wandb_run_name', default='qwen_toc_semeval')
    args = parser.parse_args()

    set_up()

    if torch.distributed.get_rank() == 0:
        wandb.init(name = args.run_name)
    trainer = Trainer(
                    task_name = args.task_name,
                    model_id = args.model_name,
                    cache_dir = args.cache_dir,
                    batch_size = 8, 
                    max_seq_len = 128,
                    max_cue_len = 64,
                    learning_rate = 0.0001,
                    weight_decay = 0.01,
                    num_epoch = 50,
                    output_path=args.output_path,
                    metric_path=args.metric_path)
    
    trainer.train()
    
    torch.distributed.destroy_process_group()