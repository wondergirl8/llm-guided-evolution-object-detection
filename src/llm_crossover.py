import os
import sys
import argparse
import random

# Ensure repo root is on sys.path so `src` imports work from generated bash scripts
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cfg.constants import *
from utils.print_utils import box_print
from candidate_policy import load_candidate_policy
from llm_utils import (split_file, submit_mixtral, submit_mixtral_hf, 
                       llm_code_qc, str2bool, extract_note, generate_augmented_code, 
                       clean_code_from_llm, retrieve_base_code, strip_leading_imports)
from src.llm_mutation import validate_dfine_proposal, validate_generated_chunk


def augment_network(input_filename_x, input_filename_y, output_filename,
                    top_p=0.15, llm_model=LLM_QWEN, temperature=0.1, apply_quality_control=False):
    """Augment Python Network Script.
    
    Parameters
    ----------
    input_filename_x : os.PathLike
        _description_
    input_filename_y : os.PathLike
        _description_
    output_filename : os.PathLike
        _description_
    top_p : float, optional
        _description_, by default 0.15
    llm_model : str, optional
        _description_, by default LLM_QWEN
    temperature : float, optional
        _description_, by default 0.1
    apply_quality_control : bool, optional
        _description_, by default False
    """
    # Split the input files
    parts_x = split_file(input_filename_x)
    parts_y = split_file(input_filename_y)
    # Create tuples of parts to be augmented. parts_x[0] is the protected
    # preamble, so indices start at 1 to address the mutable chunks.
    parts = [(x, y, idx) for idx, (x, y) in enumerate(zip(parts_x[1:], parts_y[1:]), start=1)]
    random.shuffle(parts)
    # Find differing parts
    for x, y, augment_idx in parts:
        if x.strip() != y.strip():
            break
    else:
        if not parts:
            raise ValueError('crossover has no mutable sections')
        # Legacy crossover used the last section when parents were identical.
        # Keep that behavior unless an architecture policy requests retries.
        x, y, augment_idx = random.choice(parts) if globals().get('SEED_FAMILY') == 'yolo11' and load_candidate_policy(globals()) else parts[-1]

    # Select a template file
    template_fname = random.choice(['crossover.txt', 'crossover_s.txt'])
    template_group = (globals().get('CROSSOVER_TEMPLATE_DIR', 'templates/CrossOver')
                      if globals().get('SEED_FAMILY') == 'yolo11' else 'templates/CrossOver')
    template_path = os.path.join(ROOT_DIR, template_group, template_fname)
    with open(template_path, 'r') as file:
        template_txt = file.read()

    # Add code to be augmented
    txt2llm = template_txt.format(x.strip(), y.strip())
    # Append the project's rules (shape contract, no markers, ...) as mutation does
    rules_path = globals().get("CONSTANT_RULES_PATH")
    if rules_path:
        if not os.path.isabs(rules_path):
            rules_path = os.path.join(ROOT_DIR, rules_path)
        with open(rules_path, 'r') as file:
            txt2llm = f'{txt2llm}\n{file.read()}'
    if globals().get('SEED_FAMILY') == 'dfine':
        from src.fred_dfine_proposals import generate_validated_section
        generate_validated_section(
            input_filename_x, output_filename, parts_x, augment_idx, txt2llm,
            lambda prompt: strip_leading_imports(generate_augmented_code(
                prompt, augment_idx - 1, apply_quality_control, top_p,
                llm_model, temperature)), FRED_DIR,
            attempts=globals().get('max_gen_attempts', 5), other_parent=input_filename_y)
        print('Job Done', flush=True)
        return
    if globals().get('SEED_FAMILY') == 'yolo26_p2':
        from src.fred_yolo26_proposals import generate_validated_section
        generate_validated_section(input_filename_x, output_filename, parts_x, augment_idx,
            txt2llm, lambda prompt: generate_augmented_code(prompt, augment_idx - 1,
                apply_quality_control, top_p, llm_model, temperature), FRED_DIR)
        print('Job Done', flush=True)
        return
    # Generate augmented code (retrieve_base_code indexes the mutable chunks from 0)
    policy = (None if globals().get('SEED_FAMILY') in ('dfine', 'yolo26_p2')
              else load_candidate_policy(globals()))
    attempts = globals().get('max_gen_attempts', 5) if policy else 1
    previous_error = None
    for attempt in range(attempts):
        prompt = policy.retry_prompt(txt2llm, previous_error, attempt) if policy and hasattr(policy, 'retry_prompt') else txt2llm
        retry_temperature = min(0.8, temperature + 0.1 * attempt) if policy else temperature
        retry_top_p = min(0.9, top_p + 0.1 * attempt) if policy else top_p
        code_from_llm = generate_augmented_code(prompt, augment_idx - 1, apply_quality_control,
                                                retry_top_p, llm_model, retry_temperature)
        if globals().get('SEED_FAMILY') == 'dfine':
            if not code_from_llm or code_from_llm == "ERROR":
                raise RuntimeError("ERROR proposal: LLM returned no valid code")
            code_from_llm = strip_leading_imports(code_from_llm)
            valid, reason = validate_generated_chunk(code_from_llm)
            if not valid:
                raise RuntimeError(f"ERROR proposal: {reason}")
            try:
                validate_dfine_proposal(x, code_from_llm)
                validate_dfine_proposal(y, code_from_llm)
            except ValueError as exc:
                raise RuntimeError(f"ERROR proposal: {exc}") from exc
        elif not code_from_llm or code_from_llm == "ERROR":
            previous_error = 'llm_unavailable: no usable YAML response'
            code_from_llm = x.strip()
        if policy is not None and hasattr(policy, 'repair_channel_values'):
            code_from_llm = policy.repair_channel_values(code_from_llm)
        candidate_parts = parts_x.copy()
        note_txt = extract_note(parts_x[augment_idx])
        candidate_parts[augment_idx] = f"\n{note_txt}{code_from_llm}\n"
        write_augmented_code(output_filename, candidate_parts, parts_y)
        if policy is None:
            break
        try:
            if policy.differs_from_parents(output_filename, input_filename_x, input_filename_y):
                break
            previous_error = 'unchanged_parent: built graph matches a parent'
            print(f"Crossover attempt {attempt + 1}: {previous_error}", flush=True)
        except ImportError:
            raise
        except Exception as exc:
            previous_error = f'invalid_architecture: {exc}'
            print(f"Crossover attempt {attempt + 1}: {previous_error}", flush=True)
    else:
        os.unlink(output_filename)
        raise RuntimeError(f"retry_exhausted: crossover exhausted {attempts} attempts; last={previous_error}")
    box_print(f"Python code saved to {os.path.basename(output_filename)}", print_bbox_len=120, new_line_end=False)
    print('Job done')


def write_augmented_code(output_filename, parts_x, parts_y):
    """
    Writes the augmented code to the output file.

    Parameters
    ----------
    output_filename : os.PathLike
        _description_
    parts_x : _type_
        _description_
    parts_y : _type_
        _description_
    """    

    # Only copy parent y's prompt log; without the marker, split() would return
    # y's whole preamble and paste it into the child as live code.
    marker = "# --PROMPT LOG--\n"
    if marker in parts_y[0]:
        prompt_log_cross = parts_y[0].split(marker)[0]
        prompt_log_cross = f"\n# {'='*10} Start: GeneCrossed\n{prompt_log_cross.strip()}\n# {'='*10} End:\n"
    else:
        prompt_log_cross = ""

    python_network_txt = prompt_log_cross + '# --OPTION--'.join(parts_x)

    with open(output_filename, 'w') as file:
        file.write(python_network_txt)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Augment Python Network Script.')

    # Add arguments
    parser.add_argument('input_filename_x', type=str, help='Input file name')
    parser.add_argument('input_filename_y', type=str, help='Input file name')
    parser.add_argument('output_filename', type=str, help='Output file name')
    parser.add_argument('--llm_model', type=str, default=False, help='LLM Model Name')
    parser.add_argument('--top_p', type=float, default=0.15, help='Top P value for text generation')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature value for text generation')
    parser.add_argument('--apply_quality_control', type=str2bool, default=False, help='Use LLM QC')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    augment_network(input_filename_x=args.input_filename_x,
                    input_filename_y=args.input_filename_y,
                    output_filename=args.output_filename,
                    top_p=args.top_p, 
                    llm_model=args.llm_model,
                    temperature=args.temperature,
                    apply_quality_control=args.apply_quality_control,
                   )
