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
import logging
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

import numpy as np
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
import torch
import requests
from langchain_huggingface import HuggingFaceEndpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_pipeline(token):
   

    llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    do_sample=False,
    huggingfacehub_api_token=token, 
    )
    return llm


def generate_BoC_prompt(data_point, cue_list, ablation_type = None):
    # create prompts from the loaded dataset and tokenize them
    if ablation_type == '_wo_lin':
        ablation_text = 'Do not explicitly or implicitly use any linguistic cues.'
    elif ablation_type == '_wo_con':
        ablation_text = 'Do not explicitly or implicitly use any context related cues.'
    elif ablation_type == '_wo_emo':
        ablation_text = 'Do not explicitly or implicitly use any emotion related cues.'
    else:
        ablation_text = None
    if data_point:
        return f"""
        ### Instruction:
        You are a sarcasm classification classifier. Extract the following cues: {', '.join(cue_list)} from the Input text, to assign a correct label of the Input text from ['Not Sarcastic', 'Sarcastic'].'''
        {ablation_text}

        ### Input:
        {data_point}

        ### Response: 

        ### Label: 

        """

def generate_BoC_MustARD_prompt(data_point, context, cue_list, ablation_type = None):
    # create prompts from the loaded dataset and tokenize them
    if ablation_type == '_wo_lin':
        ablation_text = 'Do not explicitly or implicitly use any linguistic cues.'
    elif ablation_type == '_wo_con':
        ablation_text = 'Do not explicitly or implicitly use any context related cues.'
    elif ablation_type == '_wo_emo':
        ablation_text = 'Do not explicitly or implicitly use any emotion related cues.'
    else:
        ablation_text = None
        
    if data_point:
        return f"""
        ### Instruction:
        You are a sarcasm classification classifier. Extract the following cues: {', '.join(cue_list)} from the Input text, to assign a correct label of the Input text from ['Not Sarcastic', 'Sarcastic'].
        {ablation_text}

        ### Context:
        {context}

        ### Input:
        {data_point}

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

def majority_vote(labels):
    count = Counter(labels)
    return count.most_common(1)[0][0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running GoC based on LLaMA for sarcasm detection.')
    parser.add_argument('--task_name', metavar='T', type=str, help='task name', default='iacv1')
    parser.add_argument('--dataset_path', metavar='F', type=str, help='dataset path', default='datasets')
    parser.add_argument('--output_path', metavar='O', type=str, help='predictions path', default='llama_output')
    parser.add_argument('--metric_path', metavar='M', type=str, help='metrics path', default='llama_output')
    parser.add_argument('--chunks', metavar='C', type=int, help='number of chunks', default=3)
    parser.add_argument('--token', metavar='K', type=str, help='token', default=None)
    parser.add_argument('--ablation_type', metavar='A', type=str, help='ablation type', default=None)
    parser.add_argument('--strategy', metavar='S', type=str, help='prompting strategy', default='boc')

    args = parser.parse_args()
    ablation_type = args.ablation_type
    task_name = args.task_name
    strategy = args.strategy
    dataset_path = f'{args.dataset_path}/test_{task_name}.csv'
    output_path = f'{args.output_path}/{strategy}/output_{strategy}_{task_name}{ablation_type}_again.csv' #f'output_toc_new/output_toc_'+ task_name +'.csv'# +'_wo_emo2.csv'
    metric_path = f'{args.metric_path}/{strategy}/metric_{strategy}_{task_name}{ablation_type}_again.json' #f'output_toc_new/metric_toc_'+ task_name +'.json'# +'_wo_emo2.json'
    token = args.token
    chunks = args.chunks

    llm = configure_pipeline(token)

    df = pd.read_csv(dataset_path, encoding_errors='ignore')
    df.dropna(inplace=True)
     
    if ablation_type == '_wo_lin':
        cue_pool = {"topic", "context", "cultural background", "common knowledge", "emotional words", "special symbols", "emotional contrasts","surface emotion"}
    elif ablation_type == '_wo_con':
        cue_pool = {"keywords", "rhetorical devices", "punctuation", "language style", "emotional words", "special symbols", "emotional contrasts","surface emotion"}
    elif ablation_type == '_wo_emo':
        cue_pool = {"keywords", "rhetorical devices", "punctuation", "language style", "topic", "context", "cultural background", "common knowledge"}
    else:
        cue_pool = {"keywords", "rhetorical devices", "punctuation", "language style", "topic", "context", "cultural background", "common knowledge", "emotional words", "special symbols", "emotional contrasts","surface emotion"}
    
    '''
        Linguistic cues: "keywords", "rhetorical devices", "punctuation", "language style"
        Contextual cues: "topic", "context", "common knowledge" 
        Emotional cues: "emotional words", "emotional contrasts"
    '''
    q=5
    num_set=3

    for i in range(num_set):
        df[f'cue_{i}'] = [get_random_cues(cue_pool, q) for _ in range(len(df))]
        # random_cues = get_random_cues(cue_pool, q)
        if task_name== 'mustard':
            df.loc[:,f'boc_prompt_{i}'] = df.apply(lambda s: generate_BoC_MustARD_prompt(s['Text'],s['Context'],s[f'cue_{i}'],ablation_type),axis=1)
        else:
            df.loc[:,f'boc_prompt_{i}'] = df.apply(lambda s: generate_BoC_prompt(s['Text'],s[f'cue_{i}'],ablation_type),axis=1)

    
    chunk_size = int(np.ceil(len(df) / chunks))
    df_chunks = []
    for chunk_num in range(chunks):
        logger.info('processing chunk {}...'.format(chunk_num))
        chunk_file_path = output_path.replace('.csv',f'_{chunk_num}.csv')
        if os.path.exists(chunk_file_path):
            df_chunk = pd.read_csv(chunk_file_path)
            df_chunks.append(df_chunk)
            continue
        df_chunk = df[chunk_num*chunk_size:min(len(df), (chunk_num+1)*chunk_size)]
        for i in range(num_set):
            labels = []
            output_texts = []
            for _, row in tqdm(df_chunk.iterrows(), total=len(df_chunk), desc=f"Processing chunk {chunk_num+1}/{chunks} for {task_name} dataset for boc task"):
                try:
                    input_prompt = row[f'boc_prompt_{i}']
                    result_prompt = PromptTemplate(template=input_prompt)
                    result_chain = LLMChain(llm=llm, prompt=result_prompt)
                    result = result_chain.run({})
                    result = result.lower().strip()
                    if re.search(r"\bnot sarcastic\b", result, re.IGNORECASE):
                        labels.append(0)
                    else:
                        labels.append(1)
                    output_texts.append(result)
                except:
                    output_texts.append("Error in generation!")
                    labels.append(0)

                
                
            df_chunk.loc[:,f'boc_output_{i}'] = output_texts
            df_chunk.loc[:,f'boc_label_{i}']= labels
        df_chunk.loc[:,'pred'] = df_chunk.apply(lambda x: majority_vote([x[f'boc_label_{i}'] for i in range(num_set)]),axis=1)
        df_chunk.to_csv(chunk_file_path, index=0)
        df_chunks.append(df_chunk)


    logger.info("Evaluation....")
    df = pd.concat(df_chunks)
    df.to_csv(output_path, index=0)
    for i in range(chunks):
        chunk_file_path = output_path.replace('.csv',f'_{i}.csv')
        if os.path.exists(chunk_file_path):
            os.remove(chunk_file_path)
    eval_performance(df['Label'], df['pred'], metric_path)


    