# llm_sarcasm_detection
Sarcasm Detection using LLMs

## Paths
**/code:** All codes to run GPT, Claude, Llama and Qwen models in main , k-shot and abalation experiments

**/datasets:** Pre-processed datasets in csv format

**/output:** Output texts and evaluations


## Download Llama model and Qwen model

```python
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct',cache_dir = 'llama/original')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', cache_dir = 'llama/original')
```

```python
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct',cache_dir = 'qwen/original')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-7B-Instruct', cache_dir = 'qwen/original')
```

## Running Llama and Qwen for Sarcasm Detection

**For io/cot/tot/coc/goc/boc methods:**
```python
python code/llama_models/llama_boc_api.py --task_name iacv2  --strategy boc
```
```python
python code/qwen_models/qwen_boc_api.py --task_name iacv2  --strategy boc
```

**For toc methods:**
```python
torchrun --nproc_per_node 6 code/llama_models/train_llama_toc_hf_ddp.py
```
```python
torchrun --nproc_per_node 6 code/qwen_models/train_qwen_toc_hf_ddp.py
```

## Running GPT and Claude for Sarcasm Detection
```python
python code/gpt_models/gpt-4o_boc.py --task_name iacv2  --strategy boc
```
```python
python code/claude_models/claude_boc.py --task_name iacv2  --strategy boc
```

## Running Llama and Claude for Ablation study
Have 3 ablation_type: _wo_lin/_wo_con/_wo_emo
**For Llama:** 
Llama with goc/boc methods:
```python
python code/llama_models/llama_boc_api.py --task_name iacv2  --strategy boc --ablation_type _wo_lin
```
For Llama with toc methods, we need to change the code in code/llama_models/toc_llama_hf.py file.

**For Claude:**
```python
python code/claude_models/claude_boc.py --task_name iacv2  --strategy boc --ablation_type _wo_lin
```

## Running GPT and Claude for Kshot study
Before running, please make sure the example file directions are correct
```python
python code/kshot_models/claude_boc_kshot.py --task_name iacv2  --strategy boc --k_fold 1
```
```python
python code/kshot_models/gpt-4o_boc_kshot.py --task_name iacv2  --strategy boc --k_fold 1
```