import os
import sys
import re
import time
import glob
import numpy as np
import transformers
import argparse
import ast
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
                       strip_leading_imports)

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


def validate_dfine_proposal(parent, proposal):
    """Require one executable class that actually changes the parent section."""
    if not proposal or proposal == "ERROR":
        raise ValueError("LLM returned no valid code")
    try:
        original = ast.parse(parent).body
        proposed = ast.parse(proposal).body
    except SyntaxError as exc:
        raise ValueError(f"invalid Python: {exc.msg}") from exc
    if (len(original) != 1 or len(proposed) != 1
            or not isinstance(original[0], ast.ClassDef)
            or not isinstance(proposed[0], ast.ClassDef)
            or proposed[0].name != original[0].name):
        raise ValueError("proposal must contain only the selected architecture class")
    if ast.dump(original[0], include_attributes=False) == ast.dump(
            proposed[0], include_attributes=False):
        raise ValueError("proposal leaves the architecture class unchanged")

def augment_network(input_filename='network.py', output_filename='network_x.py', template_txt=None,
                    top_p=0.15, llm_model=LLM_DEEPSEEK, temperature=0.1,
                    apply_quality_control=False, coverage_target=None,
                    dfine_search_space=None):
    
    print(f'Loading {input_filename} code')
    print('Using')
    dfine_search_space = dfine_search_space or globals().get('DFINE_SEARCH_SPACE', 'legacy')
    parts = split_file(input_filename)
    if coverage_target is not None:
        if globals().get('SEED_FAMILY') == 'yolo26_p2':
            from src.fred_yolo26_proposals import INITIAL_COVERAGE_TARGETS
        elif globals().get('SEED_FAMILY') == 'dfine' and dfine_search_space == 'motifs_v1':
            from src.fred_dfine_genome import INITIAL_COVERAGE_TARGETS
        else:
            raise ValueError('coverage_target requires YOLO26 or D-FINE motifs v1')
        if coverage_target not in INITIAL_COVERAGE_TARGETS:
            raise ValueError(f'unknown initial coverage target: {coverage_target}')
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
    if globals().get('SEED_FAMILY') == 'dfine' and dfine_search_space == 'motifs_v1':
        from src.fred_dfine_genome import generate_mutation
        generate_mutation(
            input_filename, output_filename, SEED_NETWORK, template_txt,
            lambda prompt: generate_augmented_code(
                prompt, 0, False, top_p, llm_model, temperature, return_raw=True),
            coverage_target=coverage_target,
            attempts=globals().get('max_gen_attempts', 5))
        print('Job Done', flush=True)
        return
    if globals().get('SEED_FAMILY') == 'dfine':
        from src.fred_dfine_proposals import generate_validated_section
        augment_idx = np.random.randint(1, len(parts))
        txt2llm = template_txt.replace("{}", parts[augment_idx].strip(), 1)
        generate_validated_section(
            input_filename, output_filename, parts, augment_idx, txt2llm,
            lambda prompt: strip_leading_imports(generate_augmented_code(
                prompt, augment_idx - 1, apply_quality_control, top_p,
                llm_model, temperature)), FRED_DIR,
            attempts=globals().get('max_gen_attempts', 5))
        print('Job Done', flush=True)
        return
    policy = (None if globals().get('SEED_FAMILY') in ('dfine', 'yolo26_p2')
              else load_candidate_policy(globals()))
    attempts = globals().get('max_gen_attempts', 5) if policy else 1
    # YOLO11 can inspect every complete YAML block in a reply. Hold its chosen
    # section fixed so feedback on a rejected proposal remains applicable.
    inspect_blocks = policy is not None and hasattr(policy, 'candidate_chunks')
    selected_idx = np.random.randint(1, len(parts)) if inspect_blocks else None
    previous_error = None
    for attempt in range(attempts):
        candidate_parts = parts.copy()
        augment_idx = (2 if coverage_target is not None else
                       selected_idx if inspect_blocks else np.random.randint(1, len(parts)))
        code2llm = parts[augment_idx]
        txt2llm = template_txt.replace("{}", code2llm.strip(), 1)
        if policy is not None and hasattr(policy, 'mutation_prompt'):
            txt2llm = policy.mutation_prompt(txt2llm, previous_error, attempt)
        elif policy is not None and hasattr(policy, 'retry_prompt'):
            txt2llm = policy.retry_prompt(txt2llm, previous_error, attempt)
        if globals().get('SEED_FAMILY') == 'yolo26_p2':
            from src.fred_yolo26_proposals import generate_validated_section
            generate_validated_section(
                input_filename, output_filename, parts, augment_idx, txt2llm,
                lambda prompt: generate_augmented_code(
                    prompt, augment_idx - 1, apply_quality_control, top_p,
                    llm_model, temperature), FRED_DIR,
                coverage_target=coverage_target)
            print('Job Done', flush=True)
            return
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
        if globals().get('SEED_FAMILY') == 'dfine':
            if not code_from_llm or code_from_llm == "ERROR":
                raise RuntimeError("ERROR proposal: LLM returned no valid code")
            code_from_llm = strip_leading_imports(code_from_llm)
            valid_code, invalid_reason = validate_generated_chunk(code_from_llm)
            if not valid_code:
                raise RuntimeError(f"ERROR proposal: {invalid_reason}")
            try:
                validate_dfine_proposal(code2llm, code_from_llm)
            except ValueError as exc:
                raise RuntimeError(f"ERROR proposal: {exc}") from exc
        if not code_from_llm or code_from_llm == "ERROR":
            previous_error = 'llm_unavailable: no usable YAML response'
            code_from_llm = code2llm.strip()
        else:
            if "dfine" in os.path.basename(input_filename) or "/dfine/" in input_filename:
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
    parser.add_argument('--coverage_target', default=None,
                        help='Initial candidate target; assigned by the controller')
    parser.add_argument('--dfine_search_space', choices=('legacy', 'motifs_v1'),
                        default=globals().get('DFINE_SEARCH_SPACE', 'legacy'))

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
                    coverage_target=args.coverage_target,
                    dfine_search_space=args.dfine_search_space,
                   )
