from langchain.chains.llm import LLMChain
import networkx as nx
import random
import pandas as pd
import json
import re
import os
from sklearn import metrics
import logging
import argparse
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd
import json
import re
import os
from sklearn import metrics
import numpy as np
from tqdm import tqdm
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_huggingface import HuggingFaceEndpoint

def configure_pipeline(token):
   

    llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    do_sample=False,
    huggingfacehub_api_token=token, 
    )
    return llm

class GoCAlgorithm(object):
    def __init__(self, text, llm, ablation_type = None):
        self.text = text
        self.llm = llm
        if ablation_type == '_wo_lin':
            print(ablation_type)
            self.cue_types = [ 
                "topic", "context", "common knowledge", 
                "emotional words", "emotional contrasts"
            ]
        elif ablation_type == '_wo_con':
            print(ablation_type)
            self.cue_types = [
                "keywords", "rhetorical devices", "punctuation", "language style", 
                "emotional words", "emotional contrasts"
            ]
        elif ablation_type == '_wo_emo':
            print(ablation_type)
            self.cue_types = [
                "keywords", "rhetorical devices", "punctuation", "language style", 
                "topic", "context", "common knowledge"
            ]
        else:
            self.cue_types = [
                "keywords", "rhetorical devices", "punctuation", "language style", 
                "topic", "context", "common knowledge", 
                "emotional words", "emotional contrasts"
            ]

        
        '''
        Linguistic cues: "keywords", "rhetorical devices", "punctuation", "language style"
        Contextual cues: "topic", "context", "common knowledge" 
        Emotional cues: "emotional words", "emotional contrasts"
        '''
        self.cues = {}
        self.graph = nx.Graph()
        self.cue_nodes = []
        self.extract_cues()
        self.construct_graph()

    def extract_cues(self):
        for cue_type in self.cue_types:
            cue_maker_template = f"Extract the brief {cue_type} information from the given text: {self.text}"
            cue_maker_prompt = PromptTemplate(template=cue_maker_template, input_variables=["cue_type", "text"])
            cue_maker_chain = LLMChain(llm=self.llm, prompt=cue_maker_prompt)
            self.cues[cue_type] = cue_maker_chain.run(cue_type=cue_type, text=self.text)

    def construct_graph(self):
        for cue_type, cue_text in self.cues.items():
            self.graph.add_node(cue_type, text=cue_text)
            self.cue_nodes.append(cue_type)
        for i in range(len(self.cue_nodes)):
            for j in range(i + 1, len(self.cue_nodes)):
                self.graph.add_edge(self.cue_nodes[i], self.cue_nodes[j])
        print("graph.nodes:", self.graph.nodes)

    def cue_evaluator(self, current_cues):
        current_cue_texts = "\n".join([self.graph.nodes[cue]['text'] for cue in current_cues])
        cue_evaluator_template = f"""
        Are the following cues sufficient to detect the sarcastic polarity of the text? Respond with 'yes' or 'no'.
        It is allowed to have at most 40% uncertainty percent on your judgement.

        ### Current cues:
        {current_cue_texts}

        ### Input:
        {self.text}

        ### Answer:
        """
        cue_evaluator_prompt = PromptTemplate(template=cue_evaluator_template, input_variables=["current_cue_texts", "text"])
        cue_evaluator_chain = LLMChain(llm=self.llm, prompt=cue_evaluator_prompt)
        evaluation = cue_evaluator_chain.run(current_cue_texts=current_cue_texts, text=self.text)
        evaluation = evaluation.strip()
        return evaluation

    def select_next_cue(self, current_cues):
        remaining_cues = list(set(self.graph.nodes) - set(current_cues))
        next_cue_template = f"""
        Suppose we have already had the Current cues: you shall select the next most promising or helpful cue from the following options: {remaining_cues} for sarcasm detection. Only return the cue without any other texts.

        ### Current cues:
        {current_cues}

        ### The Next Cue:
        """
        next_cue_prompt = PromptTemplate(template=next_cue_template, input_variables=["current_cues"])
        next_cue_chain = LLMChain(llm=self.llm, prompt=next_cue_prompt)
        next_cue_response = next_cue_chain.run(current_cues=current_cues).strip()
        print("next_cue_response:", next_cue_response)
        if next_cue_response in remaining_cues:
            return next_cue_response
        else:
            return random.choice(remaining_cues)


    def generate_result(self, current_cues):
        current_cue_texts = ";".join([self.graph.nodes[cue]['text'] for cue in current_cues])
        result_template = f"""
        You can choose to output the result directly if you believe your judgment is reliable,
        or
        Consider the current cues information, assign a correct label of the Input text from ['Not Sarcastic', 'Sarcastic'].

        ### Current cues:
        {current_cue_texts}

        ### Input:
        {self.text}

        ### Response:

        ### Label:
        """
        result_prompt = PromptTemplate(template=result_template, input_variables=["current_cue_texts", "text"])
        result_chain = LLMChain(llm=self.llm, prompt=result_prompt)
        result = result_chain.run(current_cue_texts=current_cue_texts, text=self.text)
        return result


    def detect_sarcasm(self):
        initial_cue = random.choice(self.cue_nodes)
        current_cues = [initial_cue]
        visited_nodes = set(current_cues)
        while len(visited_nodes) < len(self.cue_types): 
            evaluation = self.cue_evaluator(current_cues)
            print("evaluation:", evaluation)
            if "yes" in evaluation.lower():
                result = self.generate_result(current_cues)
                return result
          
            else:
                next_cue = self.select_next_cue(current_cues)
                if next_cue not in visited_nodes:
                    current_cues.append(next_cue)
                    visited_nodes.add(next_cue)
                else:
                    break

        result = self.generate_result(current_cues)
        return result
    
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
    parser = argparse.ArgumentParser(description='Running GoC based on LLaMA for sarcasm detection.')
    # parser.add_argument('--dataset_name', metavar='D', type=str, help='dataset name', default='iacv2')
    parser.add_argument('--dataset_path', metavar='F', type=str, help='dataset path', default='datasets')
    parser.add_argument('--task_name', metavar='T', type=str, help='task name', default='iacv2')
    parser.add_argument('--output_path', metavar='O', type=str, help='predictions path', default='llama_output')
    parser.add_argument('--metric_path', metavar='P', type=str, help='metrics path', default='llama_output')
    parser.add_argument('--chunks', metavar='C', type=int, help='number of chunks', default=3)
    parser.add_argument('--token', metavar='K', type=str, help='token', default=None)
    parser.add_argument('--ablation_type', metavar='A', type=str, help='ablation type', default=None)
    parser.add_argument('--strategy', metavar='S', type=str, help='prompting strategy', default='goc')

    args = parser.parse_args()
    ablation_type = args.ablation_type
    task_name = args.task_name
    strategy = args.strategy
    dataset_path = f'{args.dataset_path}/test_{task_name}.csv'
    output_path = f'{args.output_path}/{strategy}/output_{strategy}_{task_name}{ablation_type}.csv' #f'output_toc_new/output_toc_'+ task_name +'.csv'# +'_wo_emo2.csv'
    metric_path = f'{args.metric_path}/{strategy}/metric_{strategy}_{task_name}{ablation_type}.json' #f'output_toc_new/metric_toc_'+ task_name +'.json'# +'_wo_emo2.json'
    token = args.token
    chunks = args.chunks

    llm = configure_pipeline(token)


    df = pd.read_csv(dataset_path, encoding_errors='ignore')
    df.dropna(inplace=True)
    logger.info('generating cues...')

    chunk_size = int(np.ceil(len(df) / args.chunks))
    df_chunks = []
    for chunk_num in range(args.chunks):
        logger.info('processing chunk {}...'.format(chunk_num))
        chunk_file_path = output_path.replace('.csv',f'_{chunk_num}.csv')
        if os.path.exists(chunk_file_path):
            df_chunk = pd.read_csv(chunk_file_path)
            df_chunks.append(df_chunk)
            continue

        df_chunk = df[chunk_num*chunk_size:min(len(df), (chunk_num+1)*chunk_size)]
        output_texts = []
        labels = []
        rows_to_drop = []
        for i, row in tqdm(df_chunk.iterrows(), total=len(df_chunk), desc=f"Processing chunk {chunk_num+1}/{args.chunks} for {task_name} dataset"):
            
            goc = GoCAlgorithm(row['Text'], llm, ablation_type)
            result = goc.detect_sarcasm()
            result = result.lower().strip()
            output_texts.append(result)

            if re.search(r"\bnot sarcastic\b", result, re.IGNORECASE):
                labels.append(0)
            else:
                labels.append(1)
            print("----------------")
            
        
        df_chunk.drop(index=rows_to_drop, inplace=True)
        df_chunk.reset_index(drop=True, inplace=True)
                
        df_chunk['llm_output'] = output_texts
        df_chunk['pred']= labels
        df_chunk.to_csv(chunk_file_path, index=0)
        df_chunks.append(df_chunk)

    logger.info("Evaluation....")
    df = pd.concat(df_chunks)
    df.to_csv(output_path, index=0)
    for i in range(args.chunks):
        chunk_file_path = output_path.replace('.csv',f'_{i}.csv')
        if os.path.exists(chunk_file_path):
            os.remove(chunk_file_path)
    eval_performance(df['Label'], df['pred'], metric_path)
   