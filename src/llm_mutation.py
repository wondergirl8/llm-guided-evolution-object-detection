import os
import sys
import re
import time
import glob
import numpy as np
import transformers
import argparse
from pathlib import Path

# Ensure repo root is on sys.path so `src` imports work even when launched from nested dirs
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cfg.constants import *
from utils.print_utils import box_print
from candidate_policy import load_candidate_policy

from llm_utils import (split_file, submit_mixtral, submit_mixtral_hf, 
                       llm_code_qc, str2bool, generate_augmented_code, 
                       extract_note, clean_code_from_llm, retrieve_base_code,
                       strip_leading_imports, is_renderer_gene, RENDERER_SAMPLING)

def validate_generated_chunk(code_from_llm):
    """Reject common invalid patterns as defined by config"""
    try:
        for pattern, reason in FORBIDDEN_PATTERNS:
            if pattern in code_from_llm:
                return False, reason
        return True, ""
    except NameError:
        #if FORBIDDEN_PATTERNS not defined return true
        return True, ""

def augment_network(input_filename='network.py', output_filename='network_x.py', template_txt=None,
                    top_p=0.15, llm_model=LLM_DEEPSEEK, temperature=0.1, apply_quality_control=False):
    
    print(f'Loading {input_filename} code')
    if is_renderer_gene(input_filename):
        temperature = max(temperature, RENDERER_SAMPLING["temperature"])
        top_p = max(top_p, RENDERER_SAMPLING["top_p"])
        print(f'Renderer gene: sampling with temperature {temperature}, top_p {top_p}')
    print('Using')
    parts = split_file(input_filename)
    # prompt_templates = glob.glob(f'{ROOT_DIR}/templates/FixedPrompts/*/*.txt')
    # template_path = np.random.choice(prompt_templates)
    # template_path = f'{ROOT_DIR}/templates/{fname}'

    fname = os.path.join(ROOT_DIR, template_txt)
    with open(fname, 'r') as file:
        template_txt = file.read()
    
    # Prompt templates use a single literal "{}" marker for the code chunk.
    # Using str.format would treat JSON/dict examples in the prompt as fields.
    if "{}" not in template_txt:
        raise ValueError(f"Prompt template {fname} must contain a literal {{}} code placeholder")
    output_file = Path(output_filename)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    policy = load_candidate_policy(globals())
    attempts = globals().get('max_gen_attempts', 5) if policy else 1
    # YOLO11 can inspect every complete YAML block in a reply. Hold its chosen
    # section fixed so feedback on a rejected proposal remains applicable.
    inspect_blocks = policy is not None and hasattr(policy, 'candidate_chunks')
    selected_idx = np.random.randint(1, len(parts)) if inspect_blocks else None
    previous_error = None
    for attempt in range(attempts):
        candidate_parts = parts.copy()
        augment_idx = selected_idx if inspect_blocks else np.random.randint(1, len(parts))
        code2llm = parts[augment_idx]
        txt2llm = template_txt.replace("{}", code2llm.strip(), 1)
        if policy is not None and hasattr(policy, 'mutation_prompt'):
            txt2llm = policy.mutation_prompt(txt2llm, previous_error, attempt)
        elif policy is not None and hasattr(policy, 'retry_prompt'):
            txt2llm = policy.retry_prompt(txt2llm, previous_error, attempt)
        retry_temperature = min(0.8, temperature + 0.1 * attempt) if policy else temperature
        retry_top_p = min(0.9, top_p + 0.1 * attempt) if policy else top_p
        if inspect_blocks:
            raw_response = generate_augmented_code(
                txt2llm, augment_idx-1, apply_quality_control, retry_top_p,
                llm_model, retry_temperature, return_raw=True)
            proposals = policy.candidate_chunks(raw_response)
            if not proposals:
                previous_error = ('llm_unavailable: no response from the LLM server'
                                  if not raw_response else
                                  'invalid_architecture: no complete fenced YAML section')
                print(f"Mutation attempt {attempt + 1}: {previous_error}", flush=True)
                continue
            for block_idx, proposal in enumerate(proposals, start=1):
                valid_code, invalid_reason = validate_generated_chunk(proposal)
                if not valid_code:
                    previous_error = f'invalid_architecture: {invalid_reason}'
                else:
                    repaired = policy.repair_channel_values(proposal)
                    note_txt = extract_note(code2llm)
                    candidate_parts[augment_idx] = f"\n{note_txt}{repaired}\n"
                    output_file.write_text('# --OPTION--'.join(candidate_parts))
                    try:
                        changed_layers = policy.changed_layer_count(output_file, input_filename)
                        if changed_layers != 1:
                            raise ValueError(
                                f'mutation must change one mutable layer, changed {changed_layers}')
                        if policy.differs_from_parents(output_file, input_filename):
                            print(f"Mutation accepted YAML block {block_idx} of {len(proposals)}",
                                  flush=True)
                            break
                        previous_error = 'unchanged_parent: built graph matches the parent'
                    except ImportError:
                        raise  # Missing model packages are environment failures.
                    except Exception as exc:
                        previous_error = f'invalid_architecture: {exc}'
                print(f"Mutation attempt {attempt + 1}, YAML block {block_idx}: "
                      f"{previous_error}", flush=True)
            else:
                continue
            break
        code_from_llm = generate_augmented_code(txt2llm, augment_idx-1, apply_quality_control,
                                                retry_top_p, llm_model, retry_temperature)
        if not code_from_llm or code_from_llm == "ERROR":
            previous_error = 'llm_unavailable: no usable YAML response'
            code_from_llm = code2llm.strip()
        else:
            if any(f in os.path.basename(input_filename) or f"/{f}/" in input_filename for f in ("dfine", "renderer")):
                code_from_llm = strip_leading_imports(code_from_llm)
            valid_code, invalid_reason = validate_generated_chunk(code_from_llm)
            if not valid_code:
                print(f"Rejected generated chunk: {invalid_reason}.", flush=True)
                previous_error = f'invalid_architecture: {invalid_reason}'
                code_from_llm = code2llm.strip()
        if policy is not None and hasattr(policy, 'repair_channel_values'):
            code_from_llm = policy.repair_channel_values(code_from_llm)
        note_txt = extract_note(code2llm)
        candidate_parts[augment_idx] = f"\n{note_txt}{code_from_llm}\n"
        output_file.write_text('# --OPTION--'.join(candidate_parts))
        if policy is None:
            break
        try:
            if attempt > 0 and hasattr(policy, 'changed_layer_count'):
                changed_layers = policy.changed_layer_count(output_file, input_filename)
                if changed_layers != 1:
                    raise ValueError(f'mutation must change one mutable layer, changed {changed_layers}')
            if policy.differs_from_parents(output_file, input_filename):
                break
            previous_error = 'unchanged_parent: built graph matches the parent'
            print(f"Mutation attempt {attempt + 1}: {previous_error}", flush=True)
        except ImportError:
            raise  # Missing model packages are environment failures, not proposals to retry.
        except Exception as exc:
            previous_error = f'invalid_architecture: {exc}'
            print(f"Mutation attempt {attempt + 1}: {previous_error}", flush=True)
    else:
        output_file.unlink(missing_ok=True)
        raise RuntimeError(f"retry_exhausted: mutation exhausted {attempts} attempts; last={previous_error}")

        
    box_print(f"Python code saved to {os.path.basename(output_filename)}", print_bbox_len=120, new_line_end=False)
    print('Job Done')


    
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Augment Python Network Script.')

    # Add arguments
    parser.add_argument('input_filename', type=str, help='Input file name')
    parser.add_argument('output_filename', type=str, help='Output file name')
    parser.add_argument('template_txt', type=str, help='Template txt')
    parser.add_argument('--llm_model', type=str, default=False, help='LLM Model Name')
    parser.add_argument('--top_p', type=float, default=0.15, help='Top P value for text generation')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature value for text generation')
    parser.add_argument('--apply_quality_control', type=str2bool, default=False, help='Use LLM QC')

    # Parse the arguments
    args = parser.parse_args()
    

    # Call the function with the parsed arguments
    augment_network(input_filename=args.input_filename,
                    output_filename=args.output_filename,
                    template_txt=args.template_txt,
                    llm_model=args.llm_model,
                    top_p=args.top_p, 
                    temperature=args.temperature,
                    apply_quality_control=args.apply_quality_control,
                   )
