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
from llm_utils import (split_file, submit_mixtral, submit_mixtral_hf, 
                       llm_code_qc, str2bool, extract_note, generate_augmented_code, 
                       clean_code_from_llm, retrieve_base_code)


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

    # Select a template file
    template_fname = random.choice(['crossover.txt', 'crossover_s.txt'])
    template_path = f'{ROOT_DIR}/templates/CrossOver/{template_fname}'
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
    # Generate augmented code (retrieve_base_code indexes the mutable chunks from 0)
    code_from_llm = generate_augmented_code(txt2llm, augment_idx - 1, apply_quality_control,
                                            top_p, llm_model, temperature)
    
    if not code_from_llm:
        # Keep the first parent's chunk rather than writing the prompt into the model
        code_from_llm = x.strip()
    
    # Insert note if present
    temp_txt = parts_x[augment_idx]
    note_txt = extract_note(temp_txt)
    # Update the part with augmented code
    parts_x[augment_idx] = f"\n{note_txt}{code_from_llm}\n"
    # Prepare and write the augmented code to output file
    write_augmented_code(output_filename, parts_x, parts_y)
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
