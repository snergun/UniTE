from tqdm import tqdm
import os
import json
import time
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset
import torch
import argparse
from accelerate import Accelerator
from torch.utils.data import DataLoader
from accelerate.utils import gather_object

# --- Helper Functions (Formatting & Parsing) ---

def extract_math_answer(pred_str): # MMLU helper
    try:
        if 'boxed' in pred_str:
            ans = pred_str.split('boxed')[-1]
            flag = 1
        elif 'the answer is ' in pred_str:
            ans = pred_str.split('the answer is ')[-1].strip()
            flag = 1
        elif 'The answer is ' in pred_str:
            ans = pred_str.split('The answer is ')[-1].strip()
            flag = 1
        else:
            ans = pred_str
            flag = 0

        pattern = r'[A-D]'
        pred = re.findall(pattern, ans)

        if len(pred) >= 1:
            if flag == 0:
                pred = pred[-1] # Changed to return string instead of float for A-D
            else:
                pred = pred[0]
        else:
            pred = ""
    except Exception:
        print(f"Cannot parse the resulting num in predicted solution {pred_str}.\n")
        pred = ""
    return pred

def collate_fn(batch): # MMLU Formatting
    questions, answers = [], []
    for b in batch:
        ques = b["question"]
        A = b["A"]
        B = b["B"]
        C = b["C"]
        D = b["D"]
        # Ensure prompt variable is available in the scope or passed
        prompt_q = prompt + f'Answer the question by replying A, B, C or D.\nQuestion: {ques}\nA: {A}\nB: {B}\nC: {C}\nD: {D}\nAnswer:'
        questions.append(prompt_q)
        answers.append(b["answer"])
    return questions, answers

def parse_pred_ans(filename):
    total, correct = 0, 0
    qs = []
    try:
        with open(filename, "r", encoding="utf-8") as fr:
            for line in fr:
                jo = json.loads(line.strip())
                if jo["question"] not in qs:
                    # Simple exact match for MMLU single letter
                    correct += jo["pred"].strip() == jo["label"].strip()
                    total += 1
                    qs.append(jo["question"])
        if total > 0:
            print('num_q %d correct %d ratio %.4f' % (total, correct, float(correct / total)))
            return float(correct / total)
        return 0.0
    except FileNotFoundError:
        print("Output file not found yet.")
        return 0.0

# --- Core Token Extraction Logic ---

def get_top_k_tokens(outputs, tokenizer, k=10):
    # outputs.logits is a tuple (one per step). We take the first step [0].
    # Shape: (batch_size, vocab_size)
    logits = outputs.logits[0] 
    
    # We use the raw logits here, softmax is applied later
    probs = logits 

    top_k_indices = torch.topk(probs, k).indices
    probs = probs.tolist()

    top_k_probs = []
    # This loop handles the batch dimension (though usually batch=1 here)
    for idx, prob in zip(top_k_indices, probs):
        prob_item = []
        for i in idx:
            prob_item.append(prob[i])
        top_k_probs.append(prob_item)

    top_k_tokens = []
    for indices in top_k_indices:
        token_item = []
        for idx in indices:
            # Convert ID to token string
            token_item.append(tokenizer.convert_ids_to_tokens(idx.item(), skip_special_tokens=True))
        top_k_tokens.append(token_item)

    # Structure: List of Dictionaries (one dict per batch item)
    # Dict key: token_string, Dict value: [logit, token_id]
    v1 = []
    for token_list, prob_list, id_list in zip(top_k_tokens, top_k_probs, top_k_indices):
        item_dict = {}
        for token, prob, id_val in zip(token_list, prob_list, id_list):
            # Clean up token strings for readability
            clean_token = token.replace('▁', 'Ġ').replace('<0x0A>', '/n').replace('Ċ', '/n')
            item_dict[clean_token] = [prob, int(id_val)]
        v1.append(item_dict)

    return v1

def vocab_softmax(v1):
    # Applies Softmax to the logits contained in the v1 dictionary
    v1_new = []
    for element in v1:
        ele = {}
        ele_values = list(element.values())
        ele_logits = []
        ele_ids = []
        
        # Unpack logits and IDs
        for item in ele_values:
            ele_logits.append(item[0]) # item[0] is the logit
            ele_ids.append(item[1])    # item[1] is the id

        # Apply softmax to the logits
        softmax_probs = torch.softmax(torch.tensor(ele_logits), dim=0).tolist()
        
        # Rebuild dictionary with probabilities
        for token, prob, ids in zip(element.keys(), softmax_probs, ele_ids):
            ele[token] = [prob, ids]
        v1_new.append(ele)

    return v1_new

# --- Main Decoding Loop ---

def single_model_decoding(test_file):
    fw = open(args.output_file, "a", encoding="utf-8")

    accelerator.wait_for_everyone()
    
    # Storage for gathering
    solution_list, pred_list, label_list = [], [], []
    ori_ans_list, question_list = [], []
    top_k_info_list = [] # New list to store the probability dictionaries

    dataset = load_dataset("json", data_files=test_file)['train']
    ds_loader = DataLoader(dataset, batch_size=args.per_device_batch_size, collate_fn=collate_fn, num_workers=1)
    ds_loader = accelerator.prepare_data_loader(ds_loader)

    if accelerator.is_main_process:
        iter_item = tqdm(ds_loader)
    else:
        iter_item = ds_loader

    max_length = args.max_new_tokens # Usually 1 for MMLU

    for questions, answers in iter_item:
        output_ans = []
        
        # Tokenize
        inputs1 = tokenizer1(questions, padding=True, return_tensors="pt").to(device1)
        input_ids1 = inputs1['input_ids']
        attention_mask1 = inputs1['attention_mask']
        input_length = [len(qs) for qs in input_ids1]

        # Generate (Step-by-step loop preserved, though usually runs once)
        # Note: If max_length > 1, we only capture the probs of the *first* generated token 
        # because the assignment 'current_step_probs' would overwrite. 
        # Given MMLU is A/B/C/D, 1 token is standard.
        
        current_step_probs = []

        for i in range(max_length):
            if i == 0:
                outputs1 = model1.generate(
                    input_ids=input_ids1,
                    attention_mask=attention_mask1,
                    generation_config=generation_config1,
                )
            else:
                # If generating multiple tokens, we use past_key_values
                outputs1 = model1.generate(
                    input_ids=input_ids1,
                    attention_mask=attention_mask1,
                    past_key_values=past_key_values1,
                    generation_config=generation_config1,
                )
            
            past_key_values1 = outputs1.past_key_values

            # 1. Get Top-K Logits
            v1_logits = get_top_k_tokens(outputs1, tokenizer1, k=10)
            
            # 2. Convert to Probabilities
            v1_probs = vocab_softmax(v1_logits)
            
            # Store the probs for this step. 
            # Note: v1_probs is a list of dicts (one per batch item)
            current_step_probs = v1_probs 

            # Prepare input for next step (if max_length > 1)
            # We greedily select the top token to continue generation
            next_ids = []
            next_masks = []
            
            # Extract best token ID from the dict to append to input
            for batch_idx, token_dict in enumerate(v1_probs):
                # Find token with max prob
                best_token = max(token_dict, key=lambda k: token_dict[k][0])
                best_id = token_dict[best_token][1]
                
                # Append to history
                current_input = input_ids1[batch_idx].tolist()
                current_mask = attention_mask1[batch_idx].tolist()
                current_input.append(best_id)
                current_mask.append(1)
                
                next_ids.append(current_input)
                next_masks.append(current_mask)

            input_ids1 = torch.tensor(next_ids).to(device1)
            attention_mask1 = torch.tensor(next_masks).to(device1)

        # Decode final answer
        for qs_len, ans_ids in zip(input_length, input_ids1):
            # Decode only the new tokens
            output = tokenizer1.decode(ans_ids[qs_len:], skip_special_tokens=True)
            output = ' '.join(output.split())
            output_ans.append(output)

        # Collect results
        label_list.extend(answers)
        ori_ans_list.extend(answers)
        
        # We assume batch size aligns with v1_probs list
        top_k_info_list.extend(current_step_probs) 

        for pred_str in output_ans:
            # Clean up prompt echo if present
            if 'Question' in pred_str:
                pred_str = pred_str.split('Question:')[0].strip()
            
            print('==========output========\n', pred_str)
            solution_list.append(pred_str)
            pred_list.append(pred_str) # Simple pred, no regex extraction for now
        
        question_list.extend(questions)

    # Gather results from all GPUs (if multi-gpu)
    accelerator.wait_for_everyone()
    gather_pred = gather_object(pred_list)
    gather_label = gather_object(label_list)
    gather_solution = gather_object(solution_list)
    gather_ori_solution = gather_object(ori_ans_list)
    gather_qs = gather_object(question_list)
    gather_top_k = gather_object(top_k_info_list)

    if accelerator.is_main_process:
        for qs, pred, label, solution, ori_ans, top_k in zip(
            gather_qs, gather_pred, gather_label, gather_solution, gather_ori_solution, gather_top_k
        ):
            # Save to JSONL: Includes the 'top_k_probs' dictionary
            fw.write(json.dumps({
                "question": qs,
                "original_sln": ori_ans,
                "pred_solution": solution,
                "pred": pred,
                "label": label,
                "top_k_probs": top_k  # <--- THIS IS THE NEW DICTIONARY
            }, ensure_ascii=False) + "\n")
        fw.flush()
        fw.close()

if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--test_set", type=str, default="MMLU/test-jsonl/")
    arg_parse.add_argument("--prompts", type=str, default="MMLU/dev-jsonl/")
    arg_parse.add_argument("--model_path1", type=str, required=True, help="Your model path")
    arg_parse.add_argument("--output_file", type=str, required=True, help="Your output path")
    arg_parse.add_argument("--per_device_batch_size", type=int, default=1)
    arg_parse.add_argument("--max_new_tokens", type=int, default=1)

    args = arg_parse.parse_args()

    accelerator = Accelerator()
    device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Model: {args.model_path1}")
    model1 = AutoModelForCausalLM.from_pretrained(
        args.model_path1, 
        device_map="auto", 
        attn_implementation="sdpa",
        torch_dtype=torch.float16
    ).eval()

    tokenizer1 = AutoTokenizer.from_pretrained(args.model_path1)
    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer1.padding_side = "left"

    generation_config1 = GenerationConfig(
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer1.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        output_hidden_states=True, # Required for logits/UniTE style access
        output_scores=True,        # Required
        output_logits=True,        # Required
        return_dict_in_generate=True,
        use_cache=True,
    )

    # File setup
    test_files = [os.path.join(args.test_set, f) for f in os.listdir(args.test_set) if f.endswith('jsonl')]
    prompt_files = [os.path.join(args.prompts, f) for f in os.listdir(args.prompts) if f.endswith('jsonl')]

    acc_list = []
    
    # Map prompt files to test files
    test_file_map = {os.path.splitext(os.path.basename(f))[0]: f for f in test_files}

    for promptf in prompt_files:
        prompt_file_name = os.path.splitext(os.path.basename(promptf))[0]

        if prompt_file_name in test_file_map:
            print(f"Processing: {prompt_file_name}")
            test_file_path = test_file_map[prompt_file_name]
            
            # Load Prompt
            prompt = ''
            prompt_data = load_dataset("json", data_files=promptf)['train']
            for data in prompt_data:
                prompt += f"Question: {data['question']}\nA: {data['A']}\nB: {data['B']}\nC: {data['C']}\nD: {data['D']}\nAnswer: {data['answer']}\n\n"

            print('Start reasoning *********************')
            single_model_decoding(test_file_path)
            acc = parse_pred_ans(args.output_file)
            acc_list.append(acc)
            print('End reasoning =======================')

    if len(acc_list) > 0:
        print("The avg acc is: ", sum(acc_list) / len(acc_list))