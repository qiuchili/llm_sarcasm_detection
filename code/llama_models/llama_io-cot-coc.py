import pandas as pd
import json
import re
import os
import random
from sklearn import metrics
from collections import Counter
import argparse
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    # AutoTokenizer, 
    pipeline,
    AutoModelForCausalLM, 
    AutoTokenizer
)
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def configure_pipeline():
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct',cache_dir = 'llama/original')
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', cache_dir = 'llama/original')

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    pipe.tokenizer.pad_token_id = model.config.eos_token_id

    return pipe

def generate_IO_prompt(data_point):
    # create prompts from the loaded dataset and tokenize them
    if data_point:
        return f"""
        ### Instruction:
        '''You are a sarcasm classification classifier. Assign a correct label of the Input text from ['Not Sarcastic', 'Sarcastic']. Only return the label without any other texts.'''

        ### Input:
        {data_point}

        ### Response:

        """


def generate_CoT_prompt(data_point):
    # create prompts from the loaded dataset and tokenize them
    if data_point:
        return f"""
        ### Instruction:
        '''You are a sarcasm classification classifier. Use Chain of Thought approach to assign a correct label of the Input text from ['Not Sarcastic', 'Sarcastic'].'''

        ### Input:
        {data_point}
        Let's think step by step.

        ### Response:

        ### Label: 

        """

def generate_CoC_prompt(data_point):
    # create prompts from the loaded dataset and tokenize them
    if data_point:
        return f"""
        ### Instruction:
        '''You are a sarcasm classification classifier. Assign a correct label of the Input text from ['Not Sarcastic', 'Sarcastic'].'''

        ### Input:
        {data_point}
        You can choose to output the result directly if you believe your judgment is reliable,
        or
        You think step by step if your confidence in your judgment is less than 90%:
        Step 1: What is the SURFACE sentiment, as indicated by clues such as keywords, sentimental phrases, emojis?
        Step 2: Deduce what the sentence really means, namely the TRUE intention, by carefully checking any rhetorical devices, language style, etc.
        Step 3: Compare and analysis Step 1 and Step 2, infer the final sarcasm label.

        ### Response:

        ### Label: 

        """
    
def get_random_cues(cue_pool, n):
    return random.sample(list(cue_pool), n)


def eval_performance(y_true, y_pred, metric_path=None):

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
    try:
        roc_auc = metrics.roc_auc_score(y_true, y_pred)
        print("ROC AUC:\n\t", roc_auc) 
    except:
        print('Only one class present in y_true. ROC AUC score is not defined in that case.')
        metric_dict['ROC-AUC'] = 0

    # Confusion matrix
    print("Confusion Matrix:\n\t", metrics.confusion_matrix(y_true, y_pred))  

    if metric_path is not None:
       json.dump(metric_dict,open(metric_path,'w'),indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running io,cot or coc based on llama for sarcasm detection.')
    # parser.add_argument('--dataset_name', metavar='D', type=str, help='dataset name', default='iacv2')
    parser.add_argument('--task_name', metavar='T', type=str, help='task name', default='iacv2')
    parser.add_argument('--dataset_path', metavar='F', type=str, help='dataset path', default='datasets')
    parser.add_argument('--output_path', metavar='O', type=str, help='predictions path', default='llama_output')
    parser.add_argument('--metric_path', metavar='M', type=str, help='metrics path', default='llama_output')
    parser.add_argument('--strategy', metavar='S', type=str, help='prompting strategy', default='coc')

    args = parser.parse_args()
    task_name = args.task_name
    strategy = args.strategy
    dataset_path = f'{args.dataset_path}/test_{task_name}.csv'
    output_path = f'{args.output_path}/{strategy}/output_{strategy}_{task_name}.csv' #f'output_toc_new/output_toc_'+ task_name +'.csv'# +'_wo_emo2.csv'
    metric_path = f'{args.metric_path}/{strategy}/metric_{strategy}_{task_name}.json' #f'output_toc_new/metric_toc_'+ task_name +'.json'# +'_wo_emo2.json'
    chunks = args.chunks
    pipe = configure_pipeline()

    df = pd.read_csv(dataset_path)
    

    if strategy == 'io':
        df['prompt'] = df.apply(lambda row: [{'role': 'user','content':generate_IO_prompt(row['Text'])}], axis=1)
    elif strategy == 'cot':
        df['prompt'] = df.apply(lambda row: [{'role': 'user','content':generate_CoT_prompt(row['Text'])}], axis=1)
    elif strategy == 'coc':
        df['prompt'] = df.apply(lambda row: [{'role': 'user','content':generate_CoC_prompt(row['Text'])}], axis=1)

    else:
        print('Wrong strategy.')
        exit(1)

    dataset = Dataset.from_pandas(df)

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # for i in range(num_set):
    output_texts = []
    labels = []
    
    for out in tqdm(pipe(KeyDataset(dataset, "prompt"),  
                         batch_size=64, 
                         do_sample=True,
                         temperature=0.6,
                         top_p=0.9,
                         max_new_tokens=256,
                         eos_token_id=terminators,
                         pad_token_id=pipe.tokenizer.eos_token_id),total=len(dataset)): 

        result = out[0]['generated_text'][-1]['content']
        result = result.lower().strip()
        if re.search(r"\bnot sarcastic\b", result, re.IGNORECASE):
            labels.append(0)
        else:
            labels.append(1)
        output_texts.append(result)
        
    df['llm_output'] = output_texts
    df['pred'] = labels
    df.to_csv(output_path, index=0)
    print("Evaluation....")
    eval_performance(df['Label'], df['pred'], metric_path)
