import argparse
import sys
sys.path.append("src")
import re
import os
import glob
import time
import numpy as np
import transformers
from torch import bfloat16, float16
from utils.privit import *
from cfg.constants import *
from utils.print_utils import *



from typing import Optional
import requests
import huggingface_hub
from huggingface_hub import InferenceClient
import textwrap
from transformers import AutoTokenizer
from google import genai
from google.genai import types

def retrieve_base_code(idx):
    """Retrieves base code for quality control."""
    base_network = SEED_NETWORK
    return split_file(base_network)[1:][idx].strip()

def clean_code_from_llm(code_from_llm):
    """Cleans the code received from LLM."""
    # Take the first non-empty closed ``` block. LLMs sometimes open with a stray
    # empty fence, and a reply cut off at the token limit leaves its last block
    # unclosed; that partial code must not be used.
    blocks = re.findall(r"```[^\n`]*\n(.*?)```", code_from_llm or "", re.DOTALL)
    for block in blocks:
        if block.strip():
            return block.strip()
    # Print an error message if the code extraction fails
    print("Runtime Error: No code was generated or the format is incorrect.")
    return "ERROR"  # Return ERROR

def get_llm_code_generator(llm_model):
    # Prefer the local uvicorn-hosted model when configured
    if LOCAL_LLM:
        if llm_model in (LLM_MIXTRAL, LLM_LLAMA3, 'llama3.3'):
            llm_code_generator = submit_mixtral_local
        elif llm_model == LLM_DEEPSEEK:
            llm_code_generator = submit_deepseek_local
        else:
            print("NO LLM SPECIFIED: USING LOCAL MIXTRAL")
            llm_code_generator = submit_mixtral_local
        qc_func = llm_code_qc_hf  # uses local server for QC prompt too
    elif INFERENCE_SUBMISSION is False:
        if llm_model == LLM_MIXTRAL:
            llm_code_generator = submit_mixtral
        elif llm_model == LLM_QWEN:
            llm_code_generator = submit_qwen
        elif llm_model == LLM_GEMMA2:
            llm_code_generator = submit_gemma2
        elif llm_model == LLM_GEMMA3:
            llm_code_generator = submit_gemma3
        elif llm_model == LLM_DEEPSEEK:
            llm_code_generator = submit_deepseek
        elif llm_model == LLM_LLAMA3:
            llm_code_generator = submit_llama3_hf
        else:
            print("NO LLM SPECIFIED: USING DEEPSEEK")
            llm_code_generator = submit_deepseek
        qc_func = llm_code_qc
    else:
        if llm_model == LLM_MIXTRAL:
            llm_code_generator = submit_mixtral_hf
        elif llm_model == LLM_LLAMA3:
            llm_code_generator = submit_llama3_hf
        elif llm_model == 'gemini':
            llm_code_generator = submit_gemini_api
        elif llm_model == LLM_GEMMA2:
            llm_code_generator = submit_gemma2_hf
        else:
            print("NO LLM SPECIFIED: USING HF MIXTRAL")
            llm_code_generator = submit_mixtral_hf
        qc_func = llm_code_qc_hf
    return llm_code_generator, qc_func

def generate_augmented_code(txt2llm, augment_idx, apply_quality_control, top_p, llm_model, temperature):
    """Generates augmented code using Mixtral."""
    print("LLM being used: ", llm_model)
    box_print("PROMPT TO LLM", print_bbox_len=60, new_line_end=False)
    print(txt2llm, flush=True)
    
    llm_code_generator, qc_func = get_llm_code_generator(llm_model)
    
    if apply_quality_control:
        base_code = retrieve_base_code(augment_idx)
        code_from_llm, generate_text = llm_code_generator(txt2llm, return_gen=True, top_p=top_p, temperature=temperature)
        temp, counter = None, 0 #default to run the quality control
        while counter < max_gen_attempts and (temp not in ["NC", "OOT", "MS", "ERROR"]): #counter to deal with stubborn 
            if temp == "NC": #regenerate based on error
                prefix = "The code you generated did not contain a code output of the changes you mentioned. Make sure to include the altered code in your output.\n"
                code_from_llm, generate_text = llm_code_generator(prefix + txt2llm, return_gen=True, top_p=top_p, temperature=temperature)
            elif temp == "OOT":
                prefix = "The output you gave was cut short due to a limited number of tokens. Shorten your output to just include the altered code without the explanation.\n"
                code_from_llm, generate_text = llm_code_generator(prefix + txt2llm, return_gen=True, top_p=top_p, temperature=temperature)
            elif temp == "MS":
                prefix = "The code you generated was in multiple segments. When you output your altered code make sure it is in a single, complete code segment including the changes you made.\n"
                code_from_llm, generate_text = llm_code_generator(prefix + txt2llm, return_gen=True, top_p=top_p, temperature=temperature)
            elif temp == "ERROR": #another unforseen error
                code_from_llm, generate_text = llm_code_generator(txt2llm, return_gen=True, top_p=top_p, temperature=temperature) #just retry
            
            temp = qc_func(code_from_llm, base_code, generate_text)
            counter += 1
        
        if temp != "NC" or temp != "OOT" or temp != "MS":
            return base_code

    else:
        code_from_llm = llm_code_generator(txt2llm, top_p=top_p, temperature=temperature)
        box_print("TEXT FROM LLM", print_bbox_len=60, new_line_end=False)
        
        print(code_from_llm)

    box_print("CODE FROM LLM", print_bbox_len=60, new_line_end=False)
    code_from_llm = clean_code_from_llm(code_from_llm)

    print(code_from_llm)
    
    return code_from_llm 

def strip_leading_imports(code):
    """Drop import lines the LLM puts above a class.

    D-FINE candidates may hold only one class per section, and the protected
    preamble already imports torch, nn and F, so these lines only get the
    candidate rejected.
    """
    lines = code.splitlines()
    start = 0
    while start < len(lines) and (not lines[start].strip()
                                  or re.match(r"(import|from)\s+\S", lines[start])):
        start += 1
    return "\n".join(lines[start:])

def extract_note(txt):
    """Extracts note from the part if present."""
    if "# -- NOTE --" in txt:
        note_txt = txt.split('# -- NOTE --')
        return '# -- NOTE --\n' + note_txt[1].strip() + '# -- NOTE --\n'
    return ''

def split_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Regular expression for the pattern
    pattern = r"# --OPTION--"
    parts = re.split(pattern, content)

    return parts

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def llm_code_qc(code_from_llm, base_code, generate_text):
    # TODO: make parameter
    template_path = os.path.join(ROOT_DIR, 'templates/llm_quality_control.txt')
    with open(template_path, 'r') as file:
        template_txt = file.read()
    # add code to be augmented
    prompt2llm = template_txt.format(code_from_llm, base_code)
    print("="*120);print(prompt2llm);print("="*120)
    
    res = generate_text(prompt2llm) # clean txt
    code_from_llm = res[0]["generated_text"]
    code_from_llm = '\n'.join(code_from_llm.strip().split("```")[1].split('\n')[1:]).strip()
    return code_from_llm


def llm_code_qc_hf(code_from_llm, base_code, generate_text=None):
    # TODO: make parameter
    fname = np.random.choice(['llm_quality_control_p.txt', 'llm_quality_control_p.txt'])
    template_path = os.path.join(ROOT_DIR, f'templates/{fname}')
    with open(template_path, 'r') as file:
        template_txt = file.read()
    # add code to be augmented
    prompt2llm = template_txt.format(code_from_llm, base_code)
    box_print("QC PROMPT TO LLM", print_bbox_len=120, new_line_end=False)
    print(prompt2llm)
    
    code_from_llm = submit_mixtral_local(prompt2llm, max_new_tokens=1500, top_p=0.1, temperature=0.1, 
                      model_id="mistralai/Mixtral-8x7B-v0.1", return_gen=False)
    box_print("TEXT FROM LLM", print_bbox_len=60, new_line_end=False)
    print(code_from_llm)
    code_from_llm = clean_code_from_llm(code_from_llm)
    return code_from_llm

def submit_mixtral_hf(txt2mixtral, max_new_tokens=1024, top_p=0.15, temperature=0.1, 
                      model_id="mistralai/Mixtral-8x7B-Instruct-v0.1", return_gen=False):
    """
    This function submits a model prompt to mixtral through the HuggingFace Inference API

    Parameters
    ----------
    txt2mixtral : str
        Prompt that will be sent to mixtral
    max_new_tokens : int, optional
       A setting to tell the LLM the maximum number of tokens to return, by default 1024
    top_p : float, optional
        _description_, by default 0.15
    temperature : float, optional
        _description_, by default 0.1
    model_id : str, optional
       Which mixtral variant to utilize for inference, by default "mistralai/Mixtral-8x7B-Instruct-v0.1"
    return_gen : bool, optional
        _description_, by default False

    Returns
    -------
    str
        Model's output from inference
    """   
    max_new_tokens = np.random.randint(900, 1300)
    os.environ['HF_API_KEY'] = DONT_SCRAPE_ME
    huggingface_hub.login(new_session=False)
    client = InferenceClient(model=model_id)
    client.headers["x-use-cache"] = "0"

    instructions = [
            {
                "role": "user",
                "content": "Provide code in Python\n" + txt2mixtral,
            },     
    ]

    tokenizer_converter = AutoTokenizer.from_pretrained(model_id)
    prompt = tokenizer_converter.apply_chat_template(instructions, tokenize=False)
    results = [client.text_generation(prompt, max_new_tokens=max_new_tokens, 
                                      return_full_text=False, 
                                      temperature=temperature, seed=101)]
    if return_gen:
        return results[0], None
    else:
        return results[0]
    
def submit_gemma2_hf(txt2mixtral, max_new_tokens=1024, top_p=0.15, temperature=0.1, 
                      model_id="google/gemma-2-27b-it", return_gen=False):
    max_new_tokens = np.random.randint(900, 1300)
    os.environ['HF_API_KEY'] = DONT_SCRAPE_ME
    huggingface_hub.login(new_session=False)
    client = InferenceClient(model=model_id)
    client.headers["x-use-cache"] = "0"

    instructions = [

            {
                "role": "user",
                "content": "Provide code in Python\n" + txt2mixtral,
            },     
    ]

    tokenizer_converter = AutoTokenizer.from_pretrained(model_id)
    prompt = tokenizer_converter.apply_chat_template(instructions, tokenize=False)
    rate_limit = True
    while rate_limit :
        try:
            results = [client.text_generation(prompt, max_new_tokens=max_new_tokens, 
                                      return_full_text=False, 
                                      temperature=temperature, seed=101)]
        except Exception as e:
            print(e, flush=True)
            time.sleep(np.random.randint(80, 300))
        else:
            rate_limit = False
        
    if return_gen:
        return results[0], None
    else:
        return results[0]
    
    
def submit_llama3_hf(txt2llama, max_new_tokens=1024, top_p=0.15, temperature=0.1, 
                      model_id="meta-llama/Meta-Llama-3.1-70B-Instruct", return_gen=False):
    """
    This function submits a model prompt to Llama3 through the HuggingFace Inference API

    Parameters
    ----------
    txt2llama : str
        Prompt that will be sent to Llama3
    max_new_tokens : int, optional
        A setting to tell the LLM the maximum number of tokens to return, by default 1024
    top_p : float, optional
        _description_, by default 0.15
    temperature : float, optional
        _description_, by default 0.1
    model_id : str, optional
        Which Llama3 variant to utilize for inference, by default "meta-llama/Meta-Llama-3.1-70B-Instruct"
    return_gen : bool, optional
        _description_, by default False

    Returns
    -------
    str
        Model's output from inference
    """    
    max_new_tokens = np.random.randint(900, 1300)
    os.environ['HF_API_KEY'] = DONT_SCRAPE_ME
    huggingface_hub.login(new_session=False)
    client = InferenceClient(model=model_id)
    client.headers["x-use-cache"] = "0"

    instructions = [

            {
                "role": "user",
                "content": "Provide code in Python\n" + txt2llama,
            },     
    ]

    tokenizer_converter = AutoTokenizer.from_pretrained(model_id)
    prompt = tokenizer_converter.apply_chat_template(instructions, tokenize=False)
    results = [client.text_generation(prompt, max_new_tokens=max_new_tokens, 
                                      return_full_text=False, 
                                      temperature=temperature, seed=101)]
    if return_gen:
        return results[0], None
    else:
        return results[0]
    
def submit_mixtral(txt2mixtral, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                   model_id="mistralai/Mixtral-8x7B-Instruct-v0.1", return_gen=False):
    max_new_tokens = np.random.randint(800, 1000)
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=bfloat16,
        device_map='auto'
    )
    model.eval()
    print(model.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False, 
        task="text-generation",
        temperature=temperature, 
        top_p=top_p,  
        top_k=0, 
        max_new_tokens=max_new_tokens, 
        repetition_penalty=1.1,
        do_sample=True,
    )

    res = generate_text(txt2mixtral)
    output_txt = res[0]["generated_text"]
    box_print("LLM OUTPUT", print_bbox_len=60, new_line_end=False)
    print(output_txt)
    box_print(f'time to load in seconds: {round(time.time()-start_time)}', print_bbox_len=120, new_line_end=False)   
    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text
    
def submit_qwen(txt2qwen, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                   model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct", return_gen=False):
    max_new_tokens = np.random.randint(800, 1000)
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=bfloat16,
        device_map='auto'
    )
    model.eval()
    print(model.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,  # if using langchain set True
        task="text-generation",
        # we pass model parameters here too
        temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        top_p=top_p,  # select from top tokens whose probability add up to 15%
        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
        repetition_penalty=1.1,  # if output begins repeating increase
        do_sample=True,
    )

    res = generate_text(txt2qwen)
    output_txt = res[0]["generated_text"]
    box_print("LLM OUTPUT", print_bbox_len=60, new_line_end=False)
    print(output_txt)
    box_print(f'time to load in seconds: {round(time.time()-start_time)}', print_bbox_len=120, new_line_end=False)   
    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text
    
def submit_deepseek(txt2qwen, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                   model_id="deepseek-ai/deepseek-coder-33b-instruct", return_gen=False):
    max_new_tokens = 3000
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        #torch_dtype=float16,
        device_map='auto',
    )
    model.eval()
    print(model.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,  # if using langchain set True
        task="text-generation",
        # we pass model parameters here too
        temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        top_p=top_p,  # select from top tokens whose probability add up to 15%
        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
        repetition_penalty=1.1,  # if output begins repeating increase
        do_sample=True,
    )

    res = generate_text(txt2qwen)
    output_txt = res[0]["generated_text"]
    box_print("LLM OUTPUT", print_bbox_len=60, new_line_end=False)
    print(output_txt)
    box_print(f'time to load in seconds: {round(time.time()-start_time)}', print_bbox_len=120, new_line_end=False)   
    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text
    
def submit_gemma2(txt2gemma, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                   model_id="google/gemma-2-27b-it", return_gen=False):
#                   model_id="/home/hice1/jli3325/scratch/.cache/huggingface/hub/models--google--gemma-2-2b-it", return_gen=False):
    max_new_tokens = 3000
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
       # torch_dtype=float16,
        device_map='auto'
    )
    model.eval()
    print(model.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,  # if using langchain set True
        task="text-generation",
        # we pass model parameters here too
        temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        top_p=top_p,  # select from top tokens whose probability add up to 15%
        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
        repetition_penalty=1.1,  # if output begins repeating increase
        do_sample=True,
    )

    res = generate_text(txt2gemma)
    output_txt = res[0]["generated_text"]
    box_print("LLM OUTPUT", print_bbox_len=60, new_line_end=False)
    print(output_txt)
    box_print(f'time to load in seconds: {round(time.time()-start_time)}', print_bbox_len=120, new_line_end=False)   
    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text
    

def submit_gemma3(txt2gemma, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                   model_id="google/gemma-3-12b-it", return_gen=False):
    max_new_tokens = np.random.randint(800, 1000)
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        # torch_dtype=float16,
        device_map='auto'
    )
    model.eval()
    print(model.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,  # if using langchain set True
        task="text-generation",
        # we pass model parameters here too
        temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        top_p=top_p,  # select from top tokens whose probability add up to 15%
        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
        repetition_penalty=1.1,  # if output begins repeating increase
        do_sample=True,
    )

    res = generate_text(txt2gemma)
    output_txt = res[0]["generated_text"]
    box_print("LLM OUTPUT", print_bbox_len=60, new_line_end=False)
    print(output_txt)
    box_print(f'time to load in seconds: {round(time.time()-start_time)}', print_bbox_len=120, new_line_end=False)   
    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text
    
def get_llm_server_hostname():
    hostname = None
    hostname_file_path = HOSTNAME_DIR 
    with open(hostname_file_path, 'r') as f:
        hostname = f.readline().strip() 
    return hostname

def submit_mixtral_local(prompt, max_new_tokens=LLM_MAX_NEW_TOKENS, temperature=0.2, top_p=0.15, server_url=None, return_gen=False):
    if server_url is None:
        server_url = f"http://{os.getenv('SERVER_HOSTNAME', 'localhost')}:{PORT}/generate"
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens, # can change to random between 800 - 1000 if needed
        "temperature": temperature,
        "top_p": top_p
    }

    headers = {"Content-Type": "application/json"}

    llm_hostname = get_llm_server_hostname()
    
    print(llm_hostname)

    server_url = f"http://{llm_hostname}:{PORT}/generate"
    
    try:
        response = requests.post(server_url, headers=headers, json=payload,
                                 timeout=((10, float(os.getenv("LLMGE_LLM_HTTP_TIMEOUT_SECONDS", "600")))
                                          if globals().get('SEED_FAMILY') in ('yolo26_p2', 'dfine') else None))
        
        if response.status_code == 200:
            output_txt = response.json().get("generated_text", "No output received.")
            print(f'{response.json().get("response_time_sec", "-1")} sec')
            if return_gen is False:
                return output_txt
            else:
                return output_txt, generate_text
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def submit_deepseek_local(prompt, max_new_tokens=LLM_MAX_NEW_TOKENS, temperature=0.2, top_p=0.15, server_url=None, return_gen=False):
    if server_url is None:
        server_url = f"http://{get_llm_server_hostname()}:8000/generate"
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens, # can change to random between 800 - 1000 if needed
        "temperature": temperature,
        "top_p": top_p
    }
    print(os.getenv("SERVER_HOSTNAME", "localhost"))

    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(server_url, headers=headers, json=payload,
                                 timeout=((10, float(os.getenv("LLMGE_LLM_HTTP_TIMEOUT_SECONDS", "600")))
                                          if globals().get('SEED_FAMILY') in ('yolo26_p2', 'dfine') else None))
        
        if response.status_code == 200:
            output_txt = response.json().get("generated_text", "No output received.")
            print(f'{response.json().get("response_time_sec", "-1")} sec')
            if return_gen is False:
                return output_txt
            else:
                return output_txt, generate_text
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

    
def submit_gemini_api(txt2gemini, **kwargs):
    """
    This function submits a model prompt to Gemini through its API

    Parameters
    ----------
    txt2gemini : str
        Prompt that will be sent to Gemini

    Returns
    -------
    str
        Model's output from inference
    """   
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[txt2gemini],
        
    )
    return response.text


def mutate_prompt(llm_model, template, inference_submission=INFERENCE_SUBMISSION):
    path, filename = os.path.split(template)
    with open(template, 'r') as file:
        prompt_text = file.read()
    prompt_text = prompt_text.split("```")[0].strip()
    prompt = "Can you rephrase this text:\n```\n{}\n```".format(prompt_text)
    temp = np.random.uniform(0.1, 0.4)

    llm_code_generator, qc_func = get_llm_code_generator(llm_model)
    print("Mutating Prompts with llm:", llm_model)
    output = llm_code_generator(prompt, temperature=temp).strip()
    if "```" in output:
        output = output.split("```")[0]
    output = output + "\n```python\n{}\n```"
    with open(os.path.join(path, "mutant{}.txt".format(llm_model)), 'w') as file:
        file.write(output)
