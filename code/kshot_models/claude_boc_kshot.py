import openai
import anthropic
import pandas as pd
import json
import time
import re
import csv
import os
import random
from sklearn import metrics
import argparse
from tqdm import tqdm
import numpy as np
from collections import Counter
import numpy as np
from sklearn import metrics
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.llms import Anthropic
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.smart_llm import SmartLLMChain
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class Create_example():
    def __init__(self, example_df, k):
        self.k = k
        #self.example_df = example_df
        self.example_df_label_1 = example_df[example_df['Label'] == 1]
        self.example_df_label_0 = example_df[example_df['Label'] == 0]
            
    def forward(self, text_label):
        if text_label == 1:
            example_list = self.example_df_label_1.sample(n=self.k)['Example'].tolist()
        elif text_label == 0:
            example_list = self.example_df_label_0.sample(n=self.k)['Example'].tolist()

        output_text = "\n\n### Example:\n".join(example_list)
        return output_text


def generate_BoC_prompt(data_point, cue_list, example_text):
    # create prompts from the loaded dataset and tokenize them
    if data_point:
        return f"""
### Example:
{example_text}

————————————————————————————————————————
        
        ### Instruction:
        You are a sarcasm classification classifier. Extract the following cues: {', '.join(cue_list)} from the Input text, to assign a correct label of the Input text from ['Not Sarcastic', 'Sarcastic'].'''
        
        ### Input:
        {data_point}

        ### Brief Response: 

        ### Label: 

        """

def generate_BoC_MustARD_prompt(data_point, context, cue_list, example_text):
    # create prompts from the loaded dataset and tokenize them
    if data_point:
        return f"""
### Example:
{example_text}

————————————————————————————————————————

        ### Instruction:
        You are a sarcasm classification classifier. Extract the following cues: {', '.join(cue_list)} from the Input text, to assign a correct label of the Input text from ['Not Sarcastic', 'Sarcastic'].

        ### Context:
        {context}

        ### Input:
        {data_point}

        ### Brief Response: 

        ### Label: 

        """

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

def get_random_cues(cue_pool, n):
    return random.sample(list(cue_pool), n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running io,cot or coc based on claude for sarcasm detection.')
    parser.add_argument('--dataset_path', metavar='F', type=str, help='dataset path', default='datasets')
    parser.add_argument('--output_path', metavar='O', type=str, help='predictions path', default='claude_output')
    parser.add_argument('--metric_path', metavar='M', type=str, help='metrics path', default='claude_output')
    parser.add_argument('--task_name', metavar='T', type=str, help='predictions path', default='try')
    parser.add_argument('--strategy', metavar='S', type=str, help='strategy', default='boc')
    
    parser.add_argument('--k_shot', metavar='K', type=int, help='k shot', default=5)
    parser.add_argument('--chunks', metavar='C', type=int, help='number of chunks', default=3)
    parser.add_argument('--api_key', metavar='A', type=str, help='api key', default=None)

    args = parser.parse_args() 

    task_name = args.task_name
    strategy = args.strategy
    k = args.k_shot
    exampleset_path = f'{args.dataset_path}/claude_example_{strategy}_{task_name}.csv'
    dataset_path = f'{args.dataset_path}/test_{task_name}.csv'
    output_path = f'{args.output_path}/{strategy}/output_{strategy}_{task_name}_{k}shot.csv'
    metric_path = f'{args.metric_path}/{strategy}/metric_{strategy}_{task_name}_{k}shot.json'
    chunks = args.chunks
    api_key = args.api_key
    

    client = client = anthropic.Anthropic(api_key=api_key)

    example_df = pd.read_csv(exampleset_path, encoding_errors='ignore')
    example_df.dropna(inplace=True)

    df = pd.read_csv(dataset_path, encoding_errors='ignore')
    df.dropna(inplace=True)

    example_creator = Create_example(example_df, k)

    cue_pool = {"keywords", "rhetorical devices", "punctuation", "language style", "topic", "context", "cultural background", "common knowledge", "emotional words", "special symbols", "emotional contrasts","surface emotion"}

    q=5
    num_set=3

    df['examples'] = [example_creator.forward(int(df.iloc[index]['Label'])) for index in range(len(df))]
    for i in range(num_set):
        df[f'examples_{i}'] = [example_creator.forward(int(df.iloc[index]['Label'])) for index in range(len(df))]
        df[f'cue_{i}'] = [get_random_cues(cue_pool, q) for _ in range(len(df))]
        if task_name == 'mustard':
            df[f'boc_prompt_{i}'] = df.apply(lambda s: generate_BoC_MustARD_prompt(s['Text'],s['Context'],s[f'cue_{i}'],s[f'examples_{i}']),axis=1)
        else:
            df[f'boc_prompt_{i}'] = df.apply(lambda s: generate_BoC_prompt(s['Text'],s[f'cue_{i}'],s[f'examples_{i}']),axis=1)

    chunk_size = int(np.ceil(len(df) / chunks))
    df_chunks = []
    for chunk_num in range(chunks):
        logger.info('processing chunk {}...'.format(chunk_num))
        print(logger.info('processing chunk {}...'.format(chunk_num)))
        chunk_file_path = output_path.replace('.csv',f'_{chunk_num}.csv')
        if os.path.exists(chunk_file_path):
            df_chunk = pd.read_csv(chunk_file_path)
            df_chunks.append(df_chunk)
            continue
        df_chunk = df[chunk_num*chunk_size:min(len(df), (chunk_num+1)*chunk_size)]
        for i in range(num_set):
            labels = []
            output_texts = []
            for index, (row_id, row) in enumerate(tqdm(df_chunk.iterrows(), total=len(df_chunk), desc=f"Processing chunk {chunk_num}/{chunks} for {task_name} to get {i}-th vote for {strategy} task")):
                input_prompt = row[f'boc_prompt_{i}']

                message = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=256,
                    messages=[
                        {"role": "user", "content": input_prompt}
                    ]
                )   
                result = message.content[0].text
                result = result.lower().strip()

                if re.search(r"\bnot sarcastic\b", result, re.IGNORECASE):
                    labels.append(0)
                else:
                    labels.append(1)
                output_texts.append(result)
   
                
            df_chunk.loc[:,f'boc_output_{i}'] = output_texts
            df_chunk.loc[:,f'boc_label_{i}']= labels
        df_chunk.loc[:,'pred'] = df_chunk.apply(lambda x: majority_vote([x[f'boc_label_{i}'] for i in range(num_set)]),axis=1)
        df_chunk.to_csv(chunk_file_path, index=0)
        df_chunks.append(df_chunk)

    df = pd.concat(df_chunks)


    df.to_csv(output_path, index=0)
    for i in range(chunks):
        chunk_file_path = output_path.replace('.csv',f'_{i}.csv')
        if os.path.exists(chunk_file_path):
            os.remove(chunk_file_path)
    eval_performance(df['Label'], df['pred'], metric_path)