import random
import openai
import anthropic
import pandas as pd
import json
import time
import re
import csv
import os
import argparse
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.llms import Anthropic
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.chains import SequentialChain
from langchain_experimental.smart_llm import SmartLLMChain


label_mapping = {1: 'Sarcastic', 0: 'Not Sarcastic'}

def generate_BoC_prompt(data_point, label, cue_list):
    label = label_mapping.get(label,'Not Sarcastic')
    # create prompts from the loaded dataset and tokenize them
    if data_point:
        return f"""
        Instruction:
        Extract the following cues: {', '.join(cue_list)} from the Input text, to refine your reasoning process for assigning a {label} label to the example input.
        
        Example Input:
        {data_point}

        Example Label: {label}

        To generate the response, including the input, label, and reasoning process and conclusion.
        Breif Response:
        """

def generate_BoC_MustARD_prompt(data_point,label, context, cue_list):
    # create prompts from the loaded dataset and tokenize them
    label = label_mapping.get(label,'Not Sarcastic')
    if data_point:
        return f"""
        Instruction:
        Extract the following cues: {', '.join(cue_list)} from the Input text, to refine your reasoning process for assigning a {label} label to the example input.
        
        Example Context:
        {context}

        Example Input:
        {data_point}

        Example Label: {label}

        To generate the response, including the context, input, label, and reasoning process and conclusion.
        Breif Response:
        """

def get_random_cues(cue_pool, n):
    return random.sample(list(cue_pool), n)

def generate_CoC_prompt(data_point,label):
    # create prompts from the loaded dataset and tokenize them
    label = label_mapping.get(label,'Not Sarcastic')
    if data_point:
        return f"""
        Instruction:
        Refine your reasoning process for assigning a {label} label to the example input.

        Example Input:
        {data_point}

        Example Label: {label}

        You think step by step:
        Step 1: What is the SURFACE sentiment, as indicated by clues such as keywords, sentimental phrases, emojis?
        Step 2: Deduce what the sentence really means, namely the TRUE intention, by carefully checking any rhetorical devices, language style, etc.
        Step 3: Compare and analysis Step 1 and Step 2, infer the final sarcasm label.

        To generate the response, including the context, input, label, and reasoning process and conclusion.
        Breif Response:
        """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running io,cot or coc based on claude for sarcasm detection.')
    parser.add_argument('--dataset_path', metavar='F', type=str, help='dataset path', default='datasets')
    parser.add_argument('--task_name', metavar='T', type=str, help='predictions path', default='try')
    parser.add_argument('--strategy', metavar='S', type=str, help='strategy', default='boc')
    parser.add_argument('--api_key', metavar='A', type=str, help='api key', default=None)

    args = parser.parse_args() 

    task_name = args.task_name
    strategy = args.strategy
    exampleset_path = f'{args.dataset_path}/claude_example_{strategy}_{task_name}.csv'
    dataset_path = f'{args.dataset_path}/train_{task_name}.csv'
    if task_name == 'mustard':
            dataset_path = f'{args.dataset_path}/test_{task_name}.csv'
    api_key = args.api_key

    client = client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_csv(dataset_path, encoding_errors='ignore')
    df.dropna(inplace=True)

    cue_pool = {"keywords", "rhetorical devices", "punctuation", "language style", "topic", "context", "cultural background", "common knowledge", "emotional words", "special symbols", "emotional contrasts","surface emotion"}

    q=5

    if strategy == 'boc':
        df[f'cue'] = [get_random_cues(cue_pool, q) for _ in range(len(df))]
        if task_name == 'mustard':
            df[f'boc_prompt'] = df.apply(lambda s: generate_BoC_MustARD_prompt(s['Text'],s['Label'],s['Context'],s[f'cue']),axis=1)
        else:
            df[f'boc_prompt'] = df.apply(lambda s: generate_BoC_prompt(s['Text'],s['Label'],s[f'cue']),axis=1)

    df_list = [df[df['Label'] == 0].sample(n = 10), df[df['Label'] == 1].sample(n = 10)]

    df_example_list = [df_list[0][['Label']], df_list[1][['Label']]]

    example_list =[[],[]]
    for i, sub_df in enumerate(df_list):
        for index, (_, row) in enumerate(sub_df.iterrows()):
            if strategy == 'boc':
                input_prompt = row[f'boc_prompt']
            elif strategy == 'coc':
                input_prompt = generate_CoC_prompt(row['Text'], row['Label'])

            message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": input_prompt}
                ]
            )   
            result = message.content[0].text
            result = result.lower().strip()

            example_list[i].append(result)

        df_example_list[i].loc[:,'Example']= example_list[i]
    
    df_example = pd.concat(df_example_list, axis=0)
    df_example.reset_index(drop=True, inplace=True)

    df_example.to_csv(exampleset_path, index=0)
