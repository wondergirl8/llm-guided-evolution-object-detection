import os
import copy
import glob
import time
import json
import csv
import string
import random
import pickle
import argparse
import subprocess
import shutil
import yaml
import numpy as np
import re
import math
import ast
import urllib.request
import urllib.error
from deap import base, creator, tools
from deap.tools import HallOfFame, ParetoFront
from functools import partial
from src.utils.print_utils import print_population, print_scores, box_print, print_job_info
from src.llm_utils import split_file, retrieve_base_code
from src.candidate_policy import load_candidate_policy
from sota.FRED_LLM_GE.phase1_detection.protocol_fingerprint import verify_fingerprint
from src.cfg.constants import *
from src.cfg import constants
import glob

# Runtime-configurable prompt glob and LLM defaults
global PROMPT_GLOB
PROMPT_GLOB = PROMPTS if "PROMPTS" in globals() else "templates/Testing/Normal/**/*.txt"
DEFAULT_LLM_MODEL = globals().get("LLM_MIXTRAL", globals().get("LLM_MODEL", None))
AVAILABLE_LLM_MODELS = globals().get("ISLAND_LLMS", [])
if not AVAILABLE_LLM_MODELS and globals().get("LLM_MODEL"):
    AVAILABLE_LLM_MODELS = [globals()["LLM_MODEL"]]

LLM_SERVER_READY = False
YOLO26_SELECTION_VERSION = 'map50_95_primary_params_tiebreak_v1'
YOLO26_COVERAGE_VERSION = 'initial_p2_p5_reductions_v1'
VALIDATED_FITNESS_FAMILIES = ('yolo26_p2', 'dfine')
DFINE_MAX_INFRA_RETRIES = 1
INITIAL_POPULATION_PENDING = False
INITIAL_CREATION_PENDING = False
INITIAL_CREATION_IDS = []


def require_yolo26_selection_version(data, path):
    if SEED_FAMILY != 'yolo26_p2':
        return
    if (data.get('selection_version') != YOLO26_SELECTION_VERSION or
            data.get('coverage_version') != YOLO26_COVERAGE_VERSION):
        raise ValueError(
            f'{path} has selection/coverage versions '
            f'{data.get("selection_version")!r}/{data.get("coverage_version")!r}; '
            f'expected {YOLO26_SELECTION_VERSION!r}/{YOLO26_COVERAGE_VERSION!r}. '
            'Start a fresh YOLO26 search '
            'with separate checkpoint paths.'
        )



class ServerWindowClosed(RuntimeError):
    """The paired LLM allocation has too little time left for another operator."""


def check_server_window():
    """Use SLURM's remaining time when the launch supplies its server job ID."""
    job_id = os.getenv('LLMGE_SERVER_JOB_ID')
    if not job_id:
        return
    if not job_id.isdecimal():
        raise ValueError('LLMGE_SERVER_JOB_ID must be a numeric SLURM job ID')
    result = subprocess.run(['squeue', '-h', '-j', job_id, '-o', '%L'],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'Cannot check LLM server job {job_id}: {result.stderr.strip()}')
    remaining = result.stdout.strip()
    if not remaining:
        write_run_status({'status': 'paused_for_server_refresh', 'server_job_id': job_id,
                          'server_seconds_left': 0})
        raise ServerWindowClosed(f'LLM server job {job_id} is no longer in squeue')
    days = 0
    if '-' in remaining:
        day_text, remaining = remaining.split('-', 1)
        days = int(day_text)
    parts = remaining.split(':')
    if len(parts) != 3 or not all(part.isdecimal() for part in parts):
        raise RuntimeError(f'Unrecognized server time remaining from squeue: {remaining!r}')
    hours, minutes, seconds = map(int, parts)
    seconds_left = days * 86400 + hours * 3600 + minutes * 60 + seconds
    margin = int(os.getenv('LLMGE_MIN_SERVER_REMAINING_SECONDS', '3600'))
    if seconds_left < margin:
        write_run_status({'status': 'paused_for_server_refresh', 'server_job_id': job_id,
                          'server_seconds_left': seconds_left})
        raise ServerWindowClosed(f'LLM server job {job_id} has {seconds_left}s left; '
                                 f'require at least {margin}s before another operator')

def print_ancestry(data):
    for gene in data.keys():
        print(f'gene: {gene}')
        print(f"\t{data[gene]['GENES']}")
        print(f"\t{data[gene]['MUTATE_TYPE']}")

def load_yaml(file_path=constants.SLURM_CONFIG_DIR):
    with open(os.path.join(file_path, 'slurm_config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    return config


def fill_template_slots(template, *values):
    rendered = template
    for value in values:
        if "{}" not in rendered:
            raise ValueError("Template does not contain enough '{}' placeholders")
        rendered = rendered.replace("{}", str(value), 1)
    return rendered


def resolve_prompt_glob(prompt_group_arg):
    """Resolve the prompt glob pattern from CLI input or defaults."""
    base_glob = PROMPT_GLOB
    if prompt_group_arg:
        cleaned = prompt_group_arg.strip()
        if cleaned.startswith('templates/') or cleaned.startswith('/'):
            candidate = cleaned
        else:
            candidate = os.path.join('templates', cleaned)
        if not candidate.endswith('.txt') and '*' not in candidate:
            candidate = os.path.join(candidate, '**', '*.txt')
        return candidate
    return base_glob

def update_ancestry(gene_id_child, gene_id_parent, ancestry, mutation_type=None, gene_id_parent2=None):
    """
    Updates the ancestry data for a given child gene based on its parent(s).

    Parameters
    ----------
    gene_id_child: 
        The ID of the child gene.
    gene_id_parent: 
        The ID of the first parent gene.
    ancestry: dict 
        The global data structure for ancestry.
    mutation_type: 
        The type of mutation (for the first part of the code). Default is None.
    gene_id_parent2: 
        The ID of the second parent gene (for the second part of the code). Default is None.

    Returns
    -------
    dict
        Ancestry dictionary
    """
    # Common part for both functionalities
    ancestry[gene_id_child] = copy.deepcopy(ancestry[gene_id_parent])
    # Handle the specifics for either part 1 or part 2
    if gene_id_parent2 is None:
        # Part 1 functionality
        ancestry[gene_id_child]['GENES'] = copy.deepcopy(ancestry[gene_id_parent]['GENES']) + [gene_id_child]
        ancestry[gene_id_child]['MUTATE_TYPE'] = copy.deepcopy(ancestry[gene_id_parent]['MUTATE_TYPE']) + [mutation_type]
    else:
        # Part 2 functionality
        cross_id = f'P:{gene_id_parent2}-C:{gene_id_child}'
        ancestry[gene_id_child]['GENES'] = copy.deepcopy(ancestry[gene_id_parent]['GENES']) + [cross_id]
        ancestry[gene_id_child]['MUTATE_TYPE'] = copy.deepcopy(ancestry[gene_id_parent]['MUTATE_TYPE']) + ["CrossOver"]
    return ancestry

def generate_template(PROB_EOT, GEN_COUNT, TOP_N_GENES, SOTA_ROOT, SEED_NETWORK, ROOT_DIR, llm_model):
    """
    Generates a template based on given probabilities and gene information.
    
    Parameters
    ----------
    PROB_EOT: float
        Probability for End of Tree (EoT) operation.
    GEN_COUNT: 
        Current generation count.
    TOP_N_GENES: list
        List of top N genes.
    SOTA_ROOT: 
        Directory path for state-of-the-art root.
    SEED_NETWORK: 
        Seed network file path.
    ROOT_DIR: 
        Root directory for templates.
    
    Returns
    -------
    template_txt : str
        The template text for the mutation
    mutation_type : str
        The type of mutation generated
    """
    if (PROB_EOT > np.random.uniform()) and (GEN_COUNT > 0):
        print("\t‣ EoT")
        top_gene = np.random.choice([x[0] for x in TOP_N_GENES])
        parts_x = split_file(f"{VARIANT_DIR}/{MODEL}_{top_gene}.py")
        parts_y = split_file(SEED_NETWORK)
        parts = [(x.strip(), y.strip(), idx) for idx, (x, y) in enumerate(zip(parts_x[1:], parts_y[1:]))]
        random.shuffle(parts)
        for x, y, augment_idx in parts:
            if x.strip() != y.strip():
                break
        eot_template_path = os.path.join(ROOT_DIR, 'templates/EoT/EoT.txt')
        with open(eot_template_path, 'r') as file:
            eot_template_txt = file.read()
        template_txt = eot_template_txt.format(x, y, "{}")
        mute_type = "EoT"
    else:
        print("\t‣ Normal Prompts")
        prompt_templates = glob.glob(os.path.join(ROOT_DIR, PROMPT_GLOB), recursive=True)
        if not prompt_templates:
            raise FileNotFoundError(f"No prompt templates found with glob: {PROMPT_GLOB}")
        template_path = np.random.choice(prompt_templates)
        print(f"\t‣ Prompt template: {os.path.relpath(template_path, ROOT_DIR)}")
        mute_type = os.path.basename(template_path).split('.')[0]  # Assuming the file extension needs to be removed
        with open(template_path, 'r') as file:
            template_txt = file.read()
        rules_path = globals().get("CONSTANT_RULES_PATH", "templates/ConstantRules.txt")
        if not os.path.isabs(rules_path):
            rules_path = os.path.join(ROOT_DIR, rules_path)
        with open(rules_path, 'r') as file:
            rules_txt = file.read()
        template_txt = f'{template_txt}\n{rules_txt}'
    return template_txt, mute_type

def write_bash_script(llm_model,
                      job_name,
                      input_filename_x=f'{SOTA_ROOT}/{SEED_NETWORK}',
                      input_filename_y=None,
                      output_filename=f'{VARIANT_DIR}/{MODEL}_x.py',
                      python_file='src/llm_mutation.py', 
                      top_p=0.1, temperature=0.2,
                      gpu=None,
                      coverage_target=None,
                     ):
    
    print("WRITING write_bash_script, llm_model: ", llm_model)

    def fetch_gene(filepath):
        return os.path.basename(filepath).replace(f'{MODEL}_','').replace('.py','')
    global GLOBAL_DATA_ANCESTRY
    QC_CHECK_BOOL = PROB_QC > np.random.uniform()
    # Extract the directory path from the file path
    dir_path = os.path.dirname(output_filename)
    # Create the directory, ignore error if it already exists
    os.makedirs(dir_path, exist_ok=True)
    gene_id_parent = fetch_gene(input_filename_x)
    gene_id_child = fetch_gene(output_filename)
    if python_file=='src/llm_mutation.py':
        template_txt, mute_type = generate_template(PROB_EOT, GEN_COUNT, TOP_N_GENES, 
                                                    SOTA_ROOT, SEED_NETWORK, ROOT_DIR, llm_model)
        if GEN_COUNT >= 0: # this does not need to happen at creation of population
            GLOBAL_DATA_ANCESTRY = update_ancestry(gene_id_child, gene_id_parent, GLOBAL_DATA_ANCESTRY, 
                                                    mutation_type=mute_type, gene_id_parent2=None)
        out_dir = os.path.join(OUTPUT_DIR, str(GENERATION))
        file_path = os.path.join(out_dir, f'{gene_id_child}_model.txt')
        os.makedirs(out_dir, exist_ok=True)
        with open(file_path, 'w') as file:
            file.write(template_txt)
        temp_text = f'{python_file} {input_filename_x} {output_filename} {file_path} --top_p {top_p} --temperature {temperature}'
        python_runline = f"uv run python {temp_text} --apply_quality_control '{QC_CHECK_BOOL}' --llm_model {llm_model}"
        if coverage_target is not None:
            from src.fred_yolo26_proposals import INITIAL_COVERAGE_TARGETS
            if SEED_FAMILY != 'yolo26_p2' or coverage_target not in INITIAL_COVERAGE_TARGETS:
                raise ValueError(f'Unsupported initial coverage target: {coverage_target}')
            python_runline += f' --coverage_target {coverage_target}'
    elif python_file=='src/llm_crossover.py':
        gene_id_parent2 = fetch_gene(input_filename_y)
        GLOBAL_DATA_ANCESTRY = update_ancestry(gene_id_child, gene_id_parent, GLOBAL_DATA_ANCESTRY, 
                                                mutation_type=None, gene_id_parent2=gene_id_parent2)
        
        temp_text = f"{python_file} {input_filename_x} {input_filename_y} {output_filename} --top_p {top_p} --temperature {temperature}"
        python_runline = f"uv run python {temp_text} --apply_quality_control '{QC_CHECK_BOOL}' --llm_model {llm_model}"
    else:
        raise ValueError("Invalid python_file argument")
    config = load_yaml()
    llm_tpl = config['llm_bash_script']
    placeholder_count = llm_tpl.count("{}")

    # If the template provides a slot for GPU selection (>=2 placeholders), fill both;
    # otherwise, only fill the runline to avoid emitting the gpu_selection as a bare command.
    if len(config['gpu_selection']) > 0 and placeholder_count >= 2:
        bash_script_content = fill_template_slots(llm_tpl, config['gpu_selection'], python_runline)
    else:
        bash_script_content = fill_template_slots(llm_tpl, python_runline)
    return bash_script_content

def create_bash_file(file_path, **kwargs):
    bash_script_content = write_bash_script(**kwargs)
    # Extract the directory from the file path
    directory = os.path.dirname(file_path)
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Write the file
    with open(file_path, 'w') as file:
        file.write(bash_script_content)
    print(f"\t‣ Bash script saved to {file_path}", flush=True)


def wait_for_llm_server(timeout=3600, check_interval=10):
    """Wait for local LLM server hostname file and HTTP endpoint to become ready."""
    global LLM_SERVER_READY

    if not LOCAL_LLM:
        return True
    from src.fred_runtime import assert_llm_job_active
    assert_llm_job_active()
    if LLM_SERVER_READY:
        timeout = 15  # A previously healthy server may have reached its wall time.
        check_interval = 5

    env_timeout = os.getenv("LLM_SERVER_READY_TIMEOUT")
    env_interval = os.getenv("LLM_SERVER_READY_CHECK_INTERVAL")
    if env_timeout and not LLM_SERVER_READY:
        timeout = int(env_timeout)
    if env_interval and not LLM_SERVER_READY:
        check_interval = int(env_interval)

    start_time = time.time()
    hostname = None
    last_error = None

    while time.time() - start_time <= timeout:
        assert_llm_job_active()
        if os.path.exists(HOSTNAME_DIR):
            try:
                with open(HOSTNAME_DIR, 'r') as file:
                    hostname = file.readline().strip()
            except OSError as err:
                last_error = err

        if hostname:
            server_url = f"http://{hostname}:{PORT}/"
            try:
                with urllib.request.urlopen(server_url, timeout=5) as response:
                    if 200 <= response.status < 300:
                        LLM_SERVER_READY = True
                        print(f"\t☑ LLM server is ready at {server_url}", flush=True)
                        return True
            except urllib.error.URLError as err:
                last_error = err
            except TimeoutError as err:
                last_error = err

        elapsed = round(time.time() - start_time)
        print(
            f"\t‣ Waiting for LLM server readiness ({elapsed}s/{timeout}s). "
            f"Host file: {HOSTNAME_DIR}",
            flush=True,
        )
        time.sleep(check_interval)

    print(f"\t☠ Timed out waiting for LLM server readiness after {timeout}s", flush=True)
    LLM_SERVER_READY = False
    if last_error is not None:
        print(f"\t‣ Last readiness error: {last_error}", flush=True)
    if os.getenv('LLMGE_SERVER_JOB_ID'):
        raise RuntimeError('LLM readiness timed out; stopping this controller allocation')
    return False

def submit_bash(file_path, **kwargs):
    """ This should be general for subbing anything and returning:
        successful_sub_flag 
        job_id
    """
    if load_candidate_policy(globals()) is not None or (RUN_TAG and SEED_FAMILY in VALIDATED_FITNESS_FAMILIES):
        check_server_window()
    if not wait_for_llm_server():
        return False, None, 'llm_unavailable: server not ready'

    create_bash_file(file_path, **kwargs)
    result = subprocess.run([RUN_COMMAND, file_path], capture_output=True, text=True)
    local_output = None
    if result.returncode == 0 and LOCAL:
        local_output = result.stdout.strip()
        print("\t‣ Output:", result.stdout.strip(), flush=True)
        job_id = None
        successful_sub_flag = True
    elif result.returncode == 0:
        print("\t‣ Output:", result.stdout.strip(), flush=True)
        # print("\t‣ Script Submitted Successfully.\n\t‣ Output:", result.stdout.strip(), flush=True)
        successful_sub_flag = True
        job_id = result.stdout.split('job ')[-1].strip()
    else:
        print("\t‣ Failed to Submit Script.\n\t‣ Error:", result.stderr.strip(), flush=True)
        successful_sub_flag = False
        job_id = None
        local_output = f'operator_submission_failed: {result.stderr.strip()}'
    return successful_sub_flag, job_id, local_output

def check_contents_for_error(contents):
    """
    Checks the output of a job for any signs of error.

    Parameters:
    contents (str): output of job to check for error

    Returns:
    bool: True if job completed successfully, False if error, None if neither.  
    """
    contents_lower = contents.lower()
    if 'error infrastructure: ' in contents_lower:
        return False
    # Check for explicit error indicators
    if "traceback" in contents_lower or "slurmstepd: error" in contents_lower:
        print("\t☠ Error Found in LLM Job Output.", flush=True)
        return False
    # Check for Slurm time-limit or cancellation
    if "cancelled at" in contents_lower or "due to time limit" in contents_lower:
        print("\t☠ Job was cancelled (time limit or manual cancellation).", flush=True)
        return False
    # Success check
    if "job done" in contents_lower:
        print("\t☑ LLM Job Completed Successfully.", flush=True)
        return True
    # If Slurm Epilog has been written but the job never printed "job done",
    # the job ended abnormally (OOM, node failure, etc.)
    if "begin slurm epilog" in contents_lower:
        print("\t☠ Job ended without completion (Slurm Epilog present, no 'job done').", flush=True)
        return False
    return None
        
def check4job_completion(job_id, local_output=None, check_interval=60, timeout=3600*30, extension=""):
    """
    Check for the completion of a job by searching for its output file and scanning for errors.
    Parameters
    ----------
    job_id : str 
        The job ID to check.
    check_interval : int
        Time in seconds between checks.
    timeout: int
        Maximum time in seconds to wait for job completion.
    Returns
    -------
    state: bool
        True if job completed successfully, False otherwise
    """
    if local_output is not None:
        state = check_contents_for_error(local_output)
        if state is None:
            raise Exception('Unexpected output from job')
        else:
            return state

    start_time = time.time()
    output_file = f'{SLURM_OUTPUT_PATH}{extension}slurm-{job_id}.out'

    while True:
        # Check if the timeout is reached
        if time.time() - start_time > timeout:
            print("Timeout reached while waiting for job completion.")
            return False

        # Check if the output file exists
        if os.path.exists(output_file):
            with open(output_file, 'r') as file:
                contents = file.read()
                state = check_contents_for_error(contents)
                if state is None:
                    pass
                else:
                    return state

        # Wait for some time before checking again
        time.sleep(check_interval)
        print(f'\t‣ Waiting on check4job_completion LLM job: {job_id} Time: {round(time.time() - start_time)}s Path: {output_file}', flush=True)
        
def generate_random_string(length=20):
    # Define the characters that can be used in the string
    characters = string.ascii_letters + string.digits
    # Generate a random strconing of specified length
    random_string = ''.join(random.choice(characters) for i in range(length))
    random_string = 'xXx'+random_string
    return random_string
    
def create_individual(container, llm_model, temp_min=0.05, temp_max=0.4,
                      coverage_target=None):
    box_print("Create Individual", print_bbox_len=60, new_line_end=False)
    out_dir = os.path.join(OUTPUT_DIR, str(GENERATION))
    config = load_yaml()
    gene_id = generate_random_string(length=24)
    # Select prompte and temp
    temperature = round(random.uniform(temp_min, temp_max), 2)
    # Assign a file path and name for the model creation bash
    file_path = os.path.join(out_dir, f'{gene_id}.sh')
    successful_sub_flag, job_id, local_output = submit_bash(file_path, 
                                            input_filename_x=f'{SEED_NETWORK}',
                                            output_filename =f'{VARIANT_DIR}/{MODEL}_{gene_id}.py',
                                            gpu=config.get("LLM_GPU"),
                                            python_file='src/llm_mutation.py',
                                            llm_model=llm_model,
                                            job_name="create_individual",
                                            coverage_target=coverage_target,
                                            top_p=0.1, temperature=temperature)
    GLOBAL_DATA[gene_id] = {'sub_flag':successful_sub_flag, 'job_id':job_id, 
                            'status':'subbed file', 'fitness':None, 'start_time':time.time(),
                            'operator_type': 'creation', 'local_output': local_output,
                            'coverage_target': coverage_target}
    if SEED_FAMILY == 'dfine':
        GLOBAL_DATA[gene_id]['parent_sources'] = [SEED_NETWORK]
        if not successful_sub_flag:
            record_dfine_proposal_failure(gene_id, 'LLM job submission failed')
    GLOBAL_DATA_ANCESTRY[gene_id] = {'GENES':[gene_id], 'MUTATE_TYPE':["CREATED"]}
    
    individual = container([gene_id])  # Assign a file ID
    
    if DELAYED_CHECK:
        if successful_sub_flag or SEED_FAMILY != 'dfine':
            GLOBAL_DATA[gene_id]['status'] = 'DELAYED_CHECK'
        individual = creator.Individual([gene_id])
        if INITIAL_CREATION_PENDING and SEED_FAMILY in VALIDATED_FITNESS_FAMILIES:
            INITIAL_CREATION_IDS.append(gene_id)
            persist_initial_creation_checkpoint()
        return individual
    
    if successful_sub_flag:
        if job_id is not None:
            print(f'Checking for Job Completion: {job_id} for {gene_id}', flush=True)
        else:
            print(f'Checking completion for {gene_id}', flush=True)
        job_done = check4job_completion(job_id=job_id, local_output=local_output, extension="evolution/")
        print(f'Model Files for {gene_id} are Loaded') if job_done else print(f'Error Loading Model Files for {gene_id}', flush=True)
        if SEED_FAMILY == 'dfine' and not job_done:
            record_dfine_proposal_failure(gene_id)
    if INITIAL_CREATION_PENDING and SEED_FAMILY in VALIDATED_FITNESS_FAMILIES:
        INITIAL_CREATION_IDS.append(gene_id)
        persist_initial_creation_checkpoint()
    return individual

def submit_run(gene_id, exclude_nodes=()):
    def write_bash_script_py(gene_id, train_file=f'{TRAIN_FILE}'):
        model_file_override = RUNLINE_TMP.format(MODEL, gene_id) 
        python_runline = EVAL_RUNLINE.format(train_file, model_file_override, VARIANT_DIR=VARIANT_DIR)
        config = load_yaml()
        template = config['python_bash_script']
        if not LOCAL and globals().get('PRESERVE_SLURM_CUDA_VISIBLE_DEVICES', False):
            # The FRED GPU job must use SLURM's per-job device assignment.
            template = ''.join(
                line for line in template.splitlines(keepends=True)
                if not re.match(r'^[ \t]*(?:export[ \t]+)?CUDA_VISIBLE_DEVICES[ \t]*=', line)
            )
            python_runline = (
                'echo "Evaluation GPU binding: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} '
                'CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"\n'
                + python_runline
            )
        bash_script_content = fill_template_slots(template, python_runline)
        return bash_script_content

    # This is for subbing the python code
    def create_bash_file_py(file_path, gene_id, **kwargs):
        bash_script_content = write_bash_script_py(gene_id, **kwargs)
        with open(file_path, 'w') as file:
            file.write(bash_script_content)
        print(f"\t‣ Bash Script Saved to {file_path}")

    def submit_bash_py(file_path, gene_id, **kwargs):
        create_bash_file_py(file_path, gene_id, **kwargs)
        job_id = None
        successful_sub_flag = False
        local_output = None
        command = [RUN_COMMAND]
        if not LOCAL and globals().get('EVAL_GPU_CONSTRAINT'):
            command.extend(['-C', EVAL_GPU_CONSTRAINT])
        if not LOCAL and exclude_nodes:
            command.extend(['--exclude', ','.join(exclude_nodes)])
        command.append(file_path)
        result = subprocess.run(command, capture_output=True, text=True)
        if LOCAL:
            local_output = result.stdout.strip() + '\n' + result.stderr.strip()
            print("\t‣ Output:", local_output, flush=True)
            job_id = None
            successful_sub_flag = result.returncode == 0
        elif result.returncode == 0:
            print("\t‣ Script Submitted Successfully.\n\t‣ Output:", result.stdout.strip())
            successful_sub_flag = True
            job_id = result.stdout.split('job ')[-1].strip()
        else:
            print("\t‣ Failed to Submit script.\n\t‣ Error:", result.stderr.strip())
            successful_sub_flag = False
            job_id = None
            local_output = result.stderr.strip()
        return successful_sub_flag, job_id, local_output
    
    out_dir = os.path.join(OUTPUT_DIR, str(GENERATION))
    file_path = os.path.join(out_dir, f'{gene_id}_model.sh')
    from src.fred_session import enabled, eval_submission
    journal = (SEED_FAMILY == 'dfine' and enabled() and not LOCAL
               and CHECKPOINT_FOLDER is not None)
    if journal:
        eval_submission(CHECKPOINT_FOLDER, gene_id, state='intent',
                        retry=GLOBAL_DATA[gene_id].get('infrastructure_retries', 0) > 0)
    successful_sub_flag, job_id, local_output = submit_bash_py(file_path, gene_id)
    if journal:
        if successful_sub_flag and not str(job_id).isdecimal():
            raise RuntimeError(f'Cannot confirm evaluation job ID for {gene_id}: {job_id!r}')
        eval_submission(CHECKPOINT_FOLDER, gene_id,
                        state='submitted' if successful_sub_flag else 'failed', job_id=job_id)
    if not successful_sub_flag and SEED_FAMILY == 'dfine':
        reject_detection_fitness(gene_id, None,
                                 f'Evaluation job submission failed: {local_output}',
                                 status='FAILED: INFRASTRUCTURE')
    elif not successful_sub_flag:
        reason = ('evaluation_process_failed: local command returned an error' if LOCAL else
                  'evaluation_submission_failed: sbatch rejected the job')
        record_failure(gene_id, reason)
    if not successful_sub_flag:
        return
    GLOBAL_DATA[gene_id]['status'] = 'running eval'
    GLOBAL_DATA[gene_id]['results_job'] = job_id
    GLOBAL_DATA[gene_id]['eval_start_time'] = time.time()
    GLOBAL_DATA[gene_id]['local_output'] = local_output
    persist_initial_population_checkpoint([creator.Individual([gene_id])])
    print(f'\t‣ Running py File for {gene_id}, {job_id}')

def evalModel(individual):
    gene_id = individual[0]
    # Initially, we don't have a fitness value
    return None

def record_failure(gene_id, reason):
    """Give an unsuccessful candidate the same invalid fitness in every path."""
    entry = GLOBAL_DATA[gene_id]
    if (load_candidate_policy(globals()) is not None and
            entry.get('status') == 'completed' and
            entry.get('failure_reason') is None and
            entry.get('fitness') not in (None, INVALID_FITNESS_MAX)):
        raise RuntimeError(f"Refusing to replace successful result for {gene_id}: {reason}")
    entry.update(status='completed', fitness=INVALID_FITNESS_MAX,
                 failure_reason=str(reason), failure_category=str(reason).split(':', 1)[0])
    key = entry.get('evaluation_cache_key')
    if key and EVAL_PENDING.get(key) == gene_id:
        del EVAL_PENDING[key]
    if "RUN_CHECKPOINT_DIR" in globals():
        from sota.FRED_LLM_GE.phase1_detection.result_contract import DetectionResult, write_results
        write_results(SOTA_ROOT, gene_id, DetectionResult(*INVALID_FITNESS_MAX), reason=str(reason))


def operator_failure_reason(gene_id, operation):
    """Report the last concrete operator error and its source operation."""
    entry = GLOBAL_DATA[gene_id]
    output_file = f'{SLURM_OUTPUT_PATH}evolution/slurm-{entry.get("job_id")}.out'
    try:
        with open(output_file, encoding='utf-8', errors='replace') as stream:
            contents = stream.read()
    except OSError:
        contents = entry.get('local_output') or ''
    last_errors = re.findall(r'last=(invalid_architecture|unchanged_parent|llm_unavailable)', contents)
    if last_errors:
        category = last_errors[-1]
        entry['retry_exhausted'] = True
    elif 'Connection refused' in contents or 'ReadTimeout' in contents or 'llm_unavailable' in contents:
        category = 'llm_unavailable'
    elif not entry.get('sub_flag') and not entry.get('job_id'):
        category = 'operator_submission_failed'
    elif 'gpu_unsupported' in contents or 'not compatible with the current PyTorch installation' in contents:
        category = 'gpu_unsupported'
    elif 'unchanged_parent' in contents:
        category = 'unchanged_parent'
    elif 'invalid_architecture' in contents:
        category = 'invalid_architecture'
    else:
        category = 'retry_exhausted'
    return f'{category}: {operation}: operator job {entry.get("job_id")}'


def is_unsupported_gpu_error(message):
    """Recognize the CUDA/PyTorch mismatch seen on Blackwell allocations."""
    message = message.lower()
    return ('not compatible with the current pytorch installation' in message or
            'no kernel image is available for execution on the device' in message or
            'sm_120 is not compatible' in message)

def log_failure_counts(generation, individuals):
    """Print failures among this generation's evaluated individuals, by cause."""
    counts = {}
    for gene_id in {ind[0] for ind in individuals}:
        reason = GLOBAL_DATA.get(gene_id, {}).get('failure_reason')
        if reason:
            category = reason.split(':', 1)[0]
            counts[category] = counts.get(category, 0) + 1
    print(f"Generation {generation} failures: {sum(counts.values())} "
          f"of {len({ind[0] for ind in individuals})}; by reason: {counts}", flush=True)

def candidate_preflight(gene_id, policy):
    """Check the generated architecture before asking SLURM to train it."""
    path = f'{VARIANT_DIR}/{MODEL}_{gene_id}.py'
    try:
        architecture_hash, params = policy.built_architecture(path)
    except ImportError:
        raise  # A missing Ultralytics installation is an environment failure.
    except Exception as exc:
        record_failure(gene_id, f'invalid_architecture: {type(exc).__name__}: {exc}')
        return False
    key = policy.evaluation_cache_key(architecture_hash, run_fingerprint())
    GLOBAL_DATA[gene_id].update(architecture_hash=architecture_hash,
                                architecture_params=params, evaluation_cache_key=key)
    hit = EVAL_CACHE.get(key)
    if hit:
        fitness = tuple(hit['fitness'])
        policy.write_cached_fitness(SOTA_ROOT, gene_id, fitness)
        GLOBAL_DATA[gene_id].update(status='completed', fitness=fitness,
                                    cache_source=hit['gene_id'], failure_reason=None)
        print(f'Cached fitness for {gene_id} from {hit["gene_id"]}', flush=True)
        return False
    if key in EVAL_PENDING:
        record_failure(gene_id, f'duplicate_graph: evaluation pending for {EVAL_PENDING[key]}')
        return False
    EVAL_PENDING[key] = gene_id
    return True

def check4model2run(gene_id):
    print(f'Checking for: SOTA_ROOT {VARIANT_DIR}/{MODEL}_{gene_id}.py')
    check_exist = f'{VARIANT_DIR}/{MODEL}_{gene_id}.py'
    
    # Check if the model file exists
    if not os.path.isfile(check_exist):
        print(f'\t☠ Model file does not exist for gene_id: {gene_id}, skipping evaluation.')
        if SEED_FAMILY == 'dfine':
            record_dfine_proposal_failure(gene_id, f'Missing generated model file: {check_exist}')
        elif SEED_FAMILY == 'yolo26_p2':
            reject_detection_fitness(gene_id, None, f'Missing generated model file: {check_exist}',
                                     status='FAILED: MISSING MODEL')
        else:
            record_failure(gene_id, 'missing_model: generated model file is absent')
        return

    if SEED_FAMILY in ('yolo26_p2', 'dfine'):
        from sota.FRED_LLM_GE.phase1_detection.candidate_runner import (
            ADAPTERS, DEFAULT_PROTOCOL, FRED_DIR, load_protocol,
        )

        protocol = load_protocol(DEFAULT_PROTOCOL)
        adapter = ADAPTERS[SEED_FAMILY](protocol['families'][SEED_FAMILY], FRED_DIR)
        if SEED_FAMILY == 'dfine':
            try:
                with open(check_exist) as source:
                    candidate_ast = ast.dump(ast.parse(source.read()), include_attributes=False)
                for parent in GLOBAL_DATA[gene_id].get('parent_sources', [SEED_NETWORK]):
                    with open(parent) as source:
                        parent_ast = ast.dump(ast.parse(source.read()), include_attributes=False)
                    if candidate_ast == parent_ast:
                        record_dfine_proposal_failure(
                            gene_id, f'Generated architecture is unchanged from {parent}')
                        return
            except (OSError, SyntaxError) as exc:
                record_dfine_proposal_failure(gene_id, f'Invalid generated source: {exc}')
                return
        if SEED_FAMILY == 'yolo26_p2':
            parent_id = GLOBAL_DATA[gene_id].get('parent_gene_id')
            parent_path = (f'{VARIANT_DIR}/{MODEL}_{parent_id}.py'
                           if parent_id else SEED_NETWORK)
        try:
            if SEED_FAMILY == 'yolo26_p2':
                digest = adapter.preflight(check_exist, parent_path,
                                           stock_seed=gene_id == 'seed')
            else:
                adapter.preflight(check_exist, protocol['proxy'])
        except ImportError:
            raise  # Missing runtime dependencies are not invalid architectures.
        except Exception as exc:
            reject_detection_fitness(gene_id, None, f'Preflight: {type(exc).__name__}: {exc}',
                                     status='FAILED: PREFLIGHT')
            return
        if SEED_FAMILY == 'yolo26_p2':
            GLOBAL_DATA[gene_id]['architecture_sha256'] = digest
            previous = (find_yolo26_duplicate(gene_id, digest, adapter)
                        if gene_id != 'seed' else None)
            if previous:
                reject_detection_fitness(gene_id, None,
                                         f'Architecture duplicates {previous} ({digest})',
                                         status='FAILED: DUPLICATE ARCHITECTURE')
                return
        else:
            from src.fred_dfine_proposals import architecture_hash
            digest = architecture_hash(check_exist)
            GLOBAL_DATA[gene_id]['architecture_sha256'] = digest
            previous = next((other for records in (GLOBAL_DATA, GLOBAL_DATA_HIST)
                             for other, record in records.items()
                             if other != gene_id and record.get('architecture_sha256') == digest
                             and not record.get('failure_reason')), None)
            if previous:
                reject_detection_fitness(gene_id, None,
                                         f'Architecture duplicates {previous} ({digest})',
                                         status='FAILED: DUPLICATE ARCHITECTURE')
                return
        submit_run(gene_id)
        return

    policy = load_candidate_policy(globals())
    if policy is not None and not candidate_preflight(gene_id, policy):
        return

    submit_run(gene_id)

def evaluation_job_node(job_id):
    """Find the node of a one-node SLURM evaluation job, when available."""
    if not job_id or LOCAL:
        return None
    try:
        result = subprocess.run(
            ['sacct', '-n', '-P', '-X', '-j', str(job_id), '-o', 'NodeList'],
            capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode:
        return None
    lines = result.stdout.strip().splitlines()
    node = lines[0].split('|')[0] if lines else ''
    return node if re.fullmatch(r'[A-Za-z0-9._-]+', node) and node not in (
        'None', 'Unknown', 'Pending') else None


def retry_unavailable_gpu(gene_id, reason):
    """Retry a FRED CUDA visibility failure, excluding its node when known."""
    limit = globals().get('EVAL_GPU_RETRY_LIMIT', 0)
    entry = GLOBAL_DATA[gene_id]
    attempts = entry.get('gpu_retry_count', 0)
    if LOCAL or attempts >= limit or not reason.startswith('gpu_unavailable:'):
        return False

    old_job = entry.get('results_job')
    bad_node = evaluation_job_node(old_job)
    excluded = set(entry.get('excluded_eval_nodes', ()))
    if bad_node:
        excluded.add(bad_node)
    retry_number = attempts + 1
    results_path = f'{SOTA_ROOT}/results/{gene_id}_results.csv'
    archived_path = f'{SOTA_ROOT}/results/{gene_id}_gpu_retry_{retry_number}.csv'
    try:
        os.replace(results_path, archived_path)
    except OSError as exc:
        print(f'Cannot preserve failed GPU result for retry: {exc}', flush=True)
        return False

    entry.setdefault('gpu_retry_history', []).append({
        'job_id': old_job, 'node': bad_node, 'reason': reason,
        'result_csv': archived_path})
    entry['gpu_retry_count'] = retry_number
    entry['excluded_eval_nodes'] = sorted(excluded)
    print(f'Retrying {gene_id} after {reason}; attempt {retry_number}/{limit}; '
          f'excluding nodes {sorted(excluded)}', flush=True)
    submit_run(gene_id, exclude_nodes=sorted(excluded))
    return True



def find_yolo26_duplicate(gene_id, digest, adapter):
    """Block only seed, running, or successfully evaluated architectures."""
    if adapter.architecture_hash(SEED_NETWORK) == digest:
        return 'seed'
    for records in (GLOBAL_DATA, GLOBAL_DATA_HIST):
        for previous_id, record in records.items():
            if previous_id == gene_id or record.get('architecture_sha256') != digest:
                continue
            status = record.get('status')
            if status == 'running eval' or (
                status == 'completed'
                and normalize_detection_fitness(record.get('fitness')) != INVALID_FITNESS_MAX
            ):
                return previous_id
    return None

def normalize_detection_fitness(fitness):
    """Reject failure rows and malformed cached fitness before selection."""
    try:
        values = tuple(float(value) for value in fitness)
    except (TypeError, ValueError, OverflowError):
        return INVALID_FITNESS_MAX
    if (len(values) != 3 or not all(math.isfinite(value) for value in values)
            or not 0 <= values[0] <= 1 or not 0 <= values[1] <= 1
            or not 0 < values[2] < 999999999):
        return INVALID_FITNESS_MAX
    return values


def reject_detection_fitness(gene_id, raw_fitness, reason, status='FAILED: INVALID FITNESS'):
    """Keep the raw failure evidence while recording a non-selectable fitness."""
    record = GLOBAL_DATA[gene_id]
    record['raw_fitness'] = raw_fitness
    record['failure_reason'] = reason
    record['fitness'] = INVALID_FITNESS_MAX
    record['status'] = status
    print(f'\t☠ {SEED_FAMILY} fitness rejected for {gene_id}: {reason}; raw={raw_fitness!r}', flush=True)


def record_dfine_proposal_failure(gene_id, reason=None):
    """Keep proposal failures distinct from training and evaluation failures."""
    job_id = GLOBAL_DATA[gene_id].get('job_id')
    log_path = f'{SLURM_OUTPUT_PATH}evolution/slurm-{job_id}.out' if job_id else None
    if reason is None:
        reason = 'LLM generation job failed'
        if log_path and os.path.isfile(log_path):
            with open(log_path) as log:
                markers = [line.strip() for line in log
                           if 'ERROR proposal:' in line or
                           ('D-FINE proposal attempt ' in line and ' rejected: ' in line)]
            if markers:
                last = markers[-1]
                reason = (last.split('ERROR proposal:', 1)[1].strip()
                          if 'ERROR proposal:' in last else
                          'retry_exhausted: ' + last.split(' rejected: ', 1)[1].strip())
    if log_path:
        reason = f'{reason}; generation log: {log_path}'
    reject_detection_fitness(gene_id, None, reason, status='FAILED: PROPOSAL')


def valid_detection_population(population):
    """Return only evaluated candidates eligible for selection."""
    valid = []
    for ind in population:
        record = GLOBAL_DATA.get(ind[0], {})
        fitness = normalize_detection_fitness(ind.fitness.values)
        if fitness == INVALID_FITNESS_MAX or record.get('status', '').startswith('FAILED:'):
            print(f'\t☠ Excluding {SEED_FAMILY} candidate {ind[0]} from selection', flush=True)
            continue
        ind.fitness.values = fitness
        valid.append(ind)
    return valid


def refill_detection_population(population, llm_model):
    """Add fresh candidates without discarding valid survivors or run history."""
    global GEN_COUNT
    attempts = 0
    while len(population) < num_elites:
        if attempts >= max_gen_attempts:
            raise RuntimeError(
                f'{SEED_FAMILY} population has {len(population)} valid candidates after '
                f'{attempts} refill batches'
            )
        attempts += 1
        print(f'MINIMUM NUMBER OF IND NOT ACHIEVED, REFILL BATCH {attempts}')
        current_gen = GEN_COUNT
        GEN_COUNT = -1  # Fresh seed creation must not request parent ancestry.
        try:
            replacements = toolbox.population(n=start_population_size, llm_model=llm_model)
            delayed_creation_check(replacements)
        finally:
            GEN_COUNT = current_gen
        for ind in replacements:
            ind.fitness.values = PLACEHOLDER_FITNESS
        check_and_update_fitness(replacements)
        population.extend(valid_detection_population(replacements))
        box_print(f'CURRENT POPULATION SIZE: {len(population)}')
    return population


def dfine_infrastructure_reason(contents):
    """Accept only explicit runner or scheduler infrastructure failures."""
    from sota.FRED_LLM_GE.phase1_detection.candidate_runner import INFRASTRUCTURE_ERROR

    for line in contents.splitlines():
        line = line.strip()
        if line.startswith(INFRASTRUCTURE_ERROR):
            return line.removeprefix(INFRASTRUCTURE_ERROR)
        lower = line.lower()
        if ('slurmstepd:' in lower or 'cancelled' in lower) and (
                'due to time limit' in lower or 'node failure' in lower):
            return line
    return None


def retry_dfine_infrastructure(gene_id, reason):
    """Retry the same gene once and retain each failed job's identity."""
    record = GLOBAL_DATA[gene_id]
    record.setdefault('infrastructure_failures', []).append(
        {'job_id': record.get('results_job'), 'reason': reason})
    retries = record.get('infrastructure_retries', 0)
    if retries >= DFINE_MAX_INFRA_RETRIES:
        reject_detection_fitness(gene_id, None, reason, status='FAILED: INFRASTRUCTURE')
        return False
    record['infrastructure_retries'] = retries + 1
    submit_run(gene_id)
    return record['status'] == 'running eval'


def check4results(gene_id):
    def check4error(gene_id):
        job_id = GLOBAL_DATA[gene_id]['results_job']
        if GLOBAL_DATA[gene_id]['local_output'] is not None:
            contents = GLOBAL_DATA[gene_id]['local_output']
            state = check_contents_for_error(contents)
            if state is None:
                print(contents, flush=True)
                raise Exception('Unexpected output from job')
            else:
                return state, contents
        # there is no local output, so process with slurm
        output_file = f'{SLURM_OUTPUT_PATH}evaluation/slurm-{job_id}.out'
        print(f"Checked Path: {output_file}")
        # Check if the output file exists
        if os.path.exists(output_file):
            with open(output_file, 'r') as file:
                contents = file.read()
                state = check_contents_for_error(contents)
                if state is None:
                    pass
                else:
                    return state, contents
        return None, None
                
    job_done, job_output = check4error(gene_id)
    if SEED_FAMILY == 'dfine' and job_output:
        reason = dfine_infrastructure_reason(job_output)
        if reason:
            retry_dfine_infrastructure(gene_id, reason)
            return
    if job_done is True:
        out_dir = os.path.join(OUTPUT_DIR, str(GENERATION))
        # The job saves the model results to a file f'{gene_id}_results.csv'
        # results_path = os.path.join(out_dir, f'{gene_id}_results.csv')
        results_path = f'{SOTA_ROOT}/results/{gene_id}_results.csv'
        try:
            with open(results_path, newline='') as file:
                rows = list(csv.reader(file))
            row = rows[-1]
            if len(row) not in (len(FITNESS_WEIGHTS), len(FITNESS_WEIGHTS) + 1):
                raise ValueError('wrong number of result columns')
            fitness = tuple(float(value) for value in row[:len(FITNESS_WEIGHTS)])
            failure_reason = (row[len(FITNESS_WEIGHTS)].strip()
                              if len(row) > len(FITNESS_WEIGHTS) else '')
            if SEED_FAMILY in VALIDATED_FITNESS_FAMILIES:
                normalized = normalize_detection_fitness(fitness)
                if normalized == INVALID_FITNESS_MAX:
                    reason = failure_reason or f'Invalid result in {results_path}'
                    if SEED_FAMILY == 'dfine' and not failure_reason:
                        prefix = 'ERROR evaluating generated model: '
                        reason = next((line.removeprefix(prefix) for line in
                                       (job_output or '').splitlines() if line.startswith(prefix)),
                                      reason)
                    if retry_unavailable_gpu(gene_id, reason):
                        return
                    reject_detection_fitness(gene_id, fitness, reason)
                    return
                if failure_reason:
                    raise ValueError('successful result has a failure reason')
                fitness = normalized
            elif fitness == INVALID_FITNESS_MAX:
                reason = failure_reason or 'candidate_failed: no reason supplied'
                if is_unsupported_gpu_error(reason):
                    reason = f'gpu_unsupported: {reason}'
                if retry_unavailable_gpu(gene_id, reason):
                    return
                retries = GLOBAL_DATA[gene_id].get('gpu_retry_count', 0)
                if reason.startswith('gpu_unavailable:') and retries:
                    reason += f' (after {retries} GPU retries)'
                record_failure(gene_id, reason)
                return
            elif not all(np.isfinite(value) for value in fitness) or failure_reason:
                raise ValueError('non-finite result or failure reason outside failure sentinel')
            GLOBAL_DATA[gene_id].update(status='completed', fitness=fitness,
                                        failure_reason=None)
            key = GLOBAL_DATA[gene_id].get('evaluation_cache_key')
            if key:
                EVAL_CACHE[key] = {'fitness': fitness, 'gene_id': gene_id}
                EVAL_PENDING.pop(key, None)
        except (OSError, ValueError, IndexError, csv.Error) as exc:
            if SEED_FAMILY in VALIDATED_FITNESS_FAMILIES:
                reject_detection_fitness(gene_id, None, f'Cannot read {results_path}: {exc}',
                                         status='FAILED: RESULT FILE')
            else:
                record_failure(gene_id, f'invalid_result: {exc}')
        # print(f'Model from Gene: {gene_id} Evaluated')
    elif job_done is False:
        if SEED_FAMILY in VALIDATED_FITNESS_FAMILIES:
            reject_detection_fitness(gene_id, None, 'Evaluation job failed; see job log',
                                     status='FAILED: EVALUATION JOB')
        else:
            job_id = GLOBAL_DATA[gene_id].get('results_job')
            output_file = f'{SLURM_OUTPUT_PATH}evaluation/slurm-{job_id}.out'
            try:
                with open(output_file, encoding='utf-8', errors='replace') as stream:
                    output = stream.read()
            except OSError:
                output = GLOBAL_DATA[gene_id].get('local_output') or ''
            if is_unsupported_gpu_error(output):
                record_failure(gene_id, 'gpu_unsupported: evaluation GPU unsupported by PyTorch')
            else:
                record_failure(gene_id, 'evaluation_job_failed: SLURM or process error')
        # print(f'Model from Gene: {gene_id} Failed to Run')
    else:
        # print('Job Has Not Finished Running Yet...', flush=True)
        pass

def cancel_eval_job(gene_id, reason):
    job_id = GLOBAL_DATA[gene_id].get('results_job')
    if LOCAL or not job_id:
        return

    result = subprocess.run(['scancel', str(job_id)], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"\t‣ Cancelled eval job {job_id} for {gene_id}: {reason}", flush=True)
    else:
        stderr = result.stderr.strip()
        print(f"\t☠ Failed to cancel eval job {job_id} for {gene_id}: {stderr}", flush=True)

def check_and_update_fitness(population, timeout=EVAL_NO_PROGRESS_TIMEOUT_SECONDS, loop_delay=60):
    """ This function submits jobs and then if submitted it checks for four possibilities.
    
    timeout: (int): seconds without an evaluation result before the model job is killed
    loop_delay (int): seconds until iterating over the jobs
    
    for a job four are four possibilities:
        ‣ The Model for Gene: {gene_id} Failed to Run
        ‣ The Model for Gene: {gene_id} Completed Successfully!
        
        ‣ Still Waiting On: Gene: {gene_id}
        ‣ Timeout for gene ID {gene_id}
    """
    # Set a timeout for each job
    box_print("SUBMITTING MODELS CREATED BY LLM")
    count = 0
    while True:
        box_print(f"Checking Model Runs: {count}", print_bbox_len=60, new_line_end=False)
        all_done = True
        for ind in population:
            gene_id = ind[0]
            # check if failed job
            if gene_id not in GLOBAL_DATA:
                GLOBAL_DATA[gene_id] = {'sub_flag':False, 'job_id':'None', 'status':'completed', 
                                        'fitness':INVALID_FITNESS_MAX, 'start_time':time.time()}
            
            if GLOBAL_DATA[gene_id]['sub_flag']==False:
                if SEED_FAMILY in VALIDATED_FITNESS_FAMILIES:
                    if not GLOBAL_DATA[gene_id]['status'].startswith('FAILED:'):
                        reject_detection_fitness(gene_id, GLOBAL_DATA[gene_id].get('fitness'),
                                                 'Candidate submission failed', status='FAILED: SUBMISSION')
                    ind.fitness.values = INVALID_FITNESS_MAX
                    continue
                if not GLOBAL_DATA[gene_id].get('failure_reason'):
                    record_failure(gene_id, 'submission_failed: no job was submitted')
                ind.fitness.values = INVALID_FITNESS_MAX
                continue
            if ind.fitness.values == PLACEHOLDER_FITNESS:  # If fitness not assigned
                # check for gene_id_model.txt file
                if GLOBAL_DATA[gene_id]['status'] == 'subbed file':
                    # here we generate/sub the model file
                    # also we assign GLOBAL_DATA[gene_id]['status'] = 'running eval' in check4model2run
                    check4model2run(gene_id) 
                if GLOBAL_DATA[gene_id]['status'] == 'running eval':
                    print(f'Checking the Results for Gene: {gene_id}')
                    no_error_flag = check4results(gene_id) # here we look for the results file for gene_id
                    if no_error_flag == False:
                        print(f"LLM Failed for Gene: {gene_id}")
                        record_failure(gene_id, 'evaluation_job_failed: SLURM or process error')
                        ind.fitness.values = INVALID_FITNESS_MAX
                
                
                if GLOBAL_DATA[gene_id]['status'] == "completed":
                    # Process results and assign fitness
                    fitness_tuple = GLOBAL_DATA[gene_id]['fitness']  # Implement this function
                    if SEED_FAMILY in VALIDATED_FITNESS_FAMILIES:
                        normalized = normalize_detection_fitness(fitness_tuple)
                        if normalized == INVALID_FITNESS_MAX:
                            reject_detection_fitness(gene_id, fitness_tuple,
                                                     'Invalid cached fitness')
                        else:
                            GLOBAL_DATA[gene_id]['fitness'] = normalized
                        fitness_tuple = normalized
                    ind.fitness.values = fitness_tuple
                elif (SEED_FAMILY in VALIDATED_FITNESS_FAMILIES
                      and GLOBAL_DATA[gene_id]['status'].startswith('FAILED:')):
                    ind.fitness.values = INVALID_FITNESS_MAX
                elif time.time() - GLOBAL_DATA[gene_id].get('eval_start_time', GLOBAL_DATA[gene_id]['start_time']) > timeout:
                    print(f"Timeout for gene ID {gene_id} after {timeout} seconds without evaluation result")
                    cancel_eval_job(gene_id, f"no evaluation result for {timeout} seconds")
                    ind.fitness.values = INVALID_FITNESS_MAX
                    if SEED_FAMILY in VALIDATED_FITNESS_FAMILIES:
                        reject_detection_fitness(gene_id, None,
                                                 f'No evaluation result after {timeout}s',
                                                 status='FAILED: TIMEOUT')
                    else:
                        record_failure(gene_id, f'evaluation_timeout: no result after {timeout}s')
                else:
                    if 'results_job' not in GLOBAL_DATA[gene_id].keys():
                        record_failure(gene_id, 'submission_failed: no evaluation job id')
                        ind.fitness.values = INVALID_FITNESS_MAX
                        print(f'\t☠ No Placeholder Fitness for: {gene_id}')
                    else:
                        print(f"\t‣ Still Waiting On: Gene: {gene_id}", flush=True)
                        print_job_info(GLOBAL_DATA[gene_id])
                        all_done = False  # Some jobs are still running
        persist_initial_population_checkpoint(population)
        if all_done:
            box_print("Evalutated All Genes", print_bbox_len=60)
            break  # All jobs are done or timed out
            
        print('Delayed...', flush=True)
        time.sleep(loop_delay)  # Wait some time before checking again
        count+=1
        
def update_individual(ind, new_gene_id, old_gene_id=None, process_success=True, process_type='Mutation'):
    """
    Update an individual based on the success or failure of a process.
    
    Parameters
    ----------
    ind: 
        The individual to be updated.
    new_gene_id: 
        The new gene ID to be assigned to the individual.
    old_gene_id: optional, default=None 
        The old gene ID to be removed from GLOBAL_DATA. Optional.
    process_success: bool, optional, default=True
        Flag indicating if the process was successful. Default is True.
    process_type: str
        Type of process ('Mutation', 'Mating', etc.). Default is 'Mutation'.
    Returns
    -------
    ind:
        Updated individual
    """
    operation = 'Mutated' if process_type == 'Mutation' else 'Mated'

    if process_success:
        ind[0] = new_gene_id
        ind = creator.Individual([new_gene_id])
        if (load_candidate_policy(globals()) is None and
                SEED_FAMILY not in VALIDATED_FITNESS_FAMILIES and old_gene_id is not None and
                old_gene_id in GLOBAL_DATA):
            del GLOBAL_DATA[old_gene_id]
        print(f'\t☑ {operation}: {new_gene_id}')
    else:
        print(f'\t☠ Failed {operation}: {new_gene_id}')
        if new_gene_id in GLOBAL_DATA.keys():
            if SEED_FAMILY == 'dfine':
                GLOBAL_DATA_HIST[new_gene_id] = GLOBAL_DATA[new_gene_id].copy()
            del GLOBAL_DATA[new_gene_id]
        if old_gene_id is not None:
            ind[0] = old_gene_id
            ind = creator.Individual([old_gene_id])

    return ind

# TODO: I need to cycle through by the job id to match the sub order
def delayed_mate_check(offspring):
    if DELAYED_CHECK is True:
        for individual in offspring:
            k = individual[0]
            if k in GLOBAL_DATA and GLOBAL_DATA[k]["status"] == "DELAYED_CHECK":
                GLOBAL_DATA[k]["status"]="subbed file"
                successful_sub_flag = GLOBAL_DATA[k]["sub_flag"]
                new_gene_id, job_id = k, GLOBAL_DATA[k]["job_id"]
                if SEED_FAMILY == 'dfine' and not successful_sub_flag:
                    record_dfine_proposal_failure(new_gene_id, 'LLM job submission failed')
                print(f'Delayed Mating Check: {new_gene_id}, LLM Job ID: {job_id}')
                job_done = False
                if successful_sub_flag:
                    print(f'\t‣ Checking for Crossover Job Completion: {job_id} for {new_gene_id}')
                    job_done = check4job_completion(job_id, extension="evolution/")
                    if SEED_FAMILY == 'dfine' and not job_done:
                        record_dfine_proposal_failure(new_gene_id)

                if job_done:
                    print(f'\t‣ Model Files for {new_gene_id} are Loaded', flush=True) 
                else: 
                    print(f'\t‣ Error Loading Model Files for {new_gene_id}!!', flush=True)

                failed_process = not (successful_sub_flag and job_done)
                if failed_process and SEED_FAMILY in VALIDATED_FITNESS_FAMILIES:
                    if not GLOBAL_DATA[new_gene_id].get('failure_reason'):
                        reject_detection_fitness(
                            new_gene_id, None,
                            operator_failure_reason(new_gene_id, 'crossover'),
                            status='FAILED: OPERATOR')
                    continue
                if failed_process and load_candidate_policy(globals()) is not None:
                    record_failure(new_gene_id, operator_failure_reason(new_gene_id, 'crossover'))
                    continue
                # Same as delayed_mutate_check: on failure update_individual drops
                # the child and restores the parent.
                individual = update_individual(individual, new_gene_id, old_gene_id=LINKED_GENES[k],
                                               process_success=not failed_process, process_type='Mating')

    return offspring


def mutate_selected_offspring(offspring, llm_model):
    """Do not use failed crossover children as mutation parents."""
    skip_failed = (load_candidate_policy(globals()) is not None or
                   SEED_FAMILY in VALIDATED_FITNESS_FAMILIES)
    for mutant in offspring:
        if skip_failed:
            record = GLOBAL_DATA.get(mutant[0], {})
            if (record.get('failure_reason') or
                    (SEED_FAMILY in VALIDATED_FITNESS_FAMILIES and
                     str(record.get('status', '')).startswith('FAILED'))):
                continue
        if random.random() < mutation_probability:
            toolbox.mutate(individual=mutant, llm_model=llm_model)
            del mutant.fitness.values


def delayed_creation_check(offspring):
    if DELAYED_CHECK is True:
        for individual in offspring:
            k = individual[0]
            if k in GLOBAL_DATA and GLOBAL_DATA[k]["status"] == "DELAYED_CHECK":
                GLOBAL_DATA[k]["status"]="subbed file"
                successful_sub_flag = GLOBAL_DATA[k]["sub_flag"]
                if SEED_FAMILY == 'dfine' and not successful_sub_flag:
                    record_dfine_proposal_failure(k, 'LLM job submission failed')
                job_done = False
                if successful_sub_flag:
                    gene_id = k
                    job_id = GLOBAL_DATA[k]["job_id"]
                    print(f'Checking for Job Completion: {job_id} for {gene_id}', flush=True)
                    job_done = check4job_completion(job_id, extension="evolution/")
                    if SEED_FAMILY == 'dfine' and not job_done:
                        record_dfine_proposal_failure(gene_id)
                    elif not job_done and load_candidate_policy(globals()) is not None:
                        record_failure(gene_id, operator_failure_reason(gene_id, 'creation'))
                elif load_candidate_policy(globals()) is not None:
                    record_failure(k, operator_failure_reason(k, 'creation'))
                elif SEED_FAMILY == 'yolo26_p2':
                    reject_detection_fitness(
                        k, None, operator_failure_reason(k, 'creation'),
                        status='FAILED: OPERATOR')
                  
    return offspring

def delayed_mutate_check(offspring):
    """
    Iterates through list of offspring and checks status of running mutation jobs.
    
    Parameters
    ----------
    offspring: list
        List of offspring
    
    Returns
    -------
    offspring: list
        Updated list of offspring
    """
    if DELAYED_CHECK is True:
        for individual in offspring:
            k = individual[0]
            if k in GLOBAL_DATA and GLOBAL_DATA[k]["status"] == "DELAYED_CHECK":
                GLOBAL_DATA[k]["status"]="subbed file"
                successful_sub_flag = GLOBAL_DATA[k]["sub_flag"]
                if SEED_FAMILY == 'dfine' and not successful_sub_flag:
                    record_dfine_proposal_failure(k, 'LLM job submission failed')
                job_done = False
                if successful_sub_flag:
                    new_gene_id = k
                    job_id = GLOBAL_DATA[k]["job_id"]
                    print(f'Delayed Mutation Check: {new_gene_id}, LLM Job ID: {job_id}', flush=True)
                    print(f'\t‣ Checking for Creation Job Completion: {job_id} for {new_gene_id}')
                    job_done = check4job_completion(job_id, extension="evolution/")
                    if SEED_FAMILY == 'dfine' and not job_done:
                        record_dfine_proposal_failure(new_gene_id)
                    if job_done:
                        print(f'\t‣ Model Files for {new_gene_id} are Loaded') 
                    else: 
                        print(f'\t☠ Error Loading Model Files for {new_gene_id}')

                    failed_process = not (successful_sub_flag and job_done)
                    old_gene_id = LINKED_GENES[k]
                    if failed_process and SEED_FAMILY in VALIDATED_FITNESS_FAMILIES:
                        if not GLOBAL_DATA[new_gene_id].get('failure_reason'):
                            reject_detection_fitness(
                                new_gene_id, None,
                                operator_failure_reason(new_gene_id, 'mutation'),
                                status='FAILED: OPERATOR')
                        continue
                    if failed_process and load_candidate_policy(globals()) is not None:
                        record_failure(new_gene_id, operator_failure_reason(new_gene_id, 'mutation'))
                        continue
                    individual = update_individual(individual, new_gene_id, old_gene_id=old_gene_id,
                                                   process_success=not failed_process, process_type='Mutation')
                elif load_candidate_policy(globals()) is not None:
                    record_failure(k, operator_failure_reason(k, 'mutation'))
                elif SEED_FAMILY == 'yolo26_p2':
                    reject_detection_fitness(
                        k, None, operator_failure_reason(k, 'mutation'),
                        status='FAILED: OPERATOR')
                  
    return offspring

def customCrossover(ind1, ind2, llm_model):
    def combine_elements(ind1, ind2, llm_model, temp_min=0.05, temp_max=0.1):
        """
        Combine elements of two individuals to create a new individual.
        Parameters:
        ind1, ind2 (list): The parent individuals.
        Returns:
        str: The gene ID of the new individual.
        """
        global GLOBAL_DATA
        out_dir = os.path.join(OUTPUT_DIR, str(GENERATION))
        config = load_yaml()
        # Retrieve gene IDs from the individuals
        gene_id_1 = ind1[0]
        gene_id_2 = ind2[0]
        # Generate the crossover query
        print(f'Mating: {gene_id_1} and {gene_id_2}')
        temperature = round(random.uniform(temp_min, temp_max), 2)
        # Generate a new gene ID for the offspring
        new_gene_id = generate_random_string(length=24)
        # Create the bash file for the new job
        file_path = os.path.join(out_dir, f'{new_gene_id}.sh')
        successful_sub_flag, job_id, local_output = submit_bash(file_path, 
                                          input_filename_x=f'{VARIANT_DIR}/{MODEL}_{gene_id_1}.py',
                                          input_filename_y=f'{VARIANT_DIR}/{MODEL}_{gene_id_2}.py',
                                          output_filename=f'{VARIANT_DIR}/{MODEL}_{new_gene_id}.py',
                                          gpu=config.get("LLM_GPU"),
                                          python_file='src/llm_crossover.py',
                                          job_name="crossover_operation",
                                          top_p=0.1, llm_model=llm_model, temperature=temperature)

        # Update global data for the new individual
        GLOBAL_DATA[new_gene_id] = {'sub_flag':successful_sub_flag, 'job_id':job_id, 
                                    'status':'subbed file', 'fitness':None, 'start_time':time.time(),
                                    'operator_type': 'crossover', 'local_output': local_output}
        if SEED_FAMILY == 'dfine':
            GLOBAL_DATA[new_gene_id]['parent_sources'] = [
                f'{VARIANT_DIR}/{MODEL}_{gene_id_1}.py',
                f'{VARIANT_DIR}/{MODEL}_{gene_id_2}.py',
            ]
        if SEED_FAMILY == 'yolo26_p2':
            GLOBAL_DATA[new_gene_id]['parent_gene_id'] = gene_id_1
        
        if DELAYED_CHECK:
            GLOBAL_DATA[new_gene_id]['status'] = 'DELAYED_CHECK'
            return new_gene_id, None
        
        if successful_sub_flag:
            print(f'\t‣ Checking for Crossover Job Completion: {job_id} for {new_gene_id}')
            job_done = check4job_completion(job_id, local_output, extension="evolution/")
            if SEED_FAMILY == 'dfine' and not job_done:
                record_dfine_proposal_failure(new_gene_id)
            if job_done:
                print(f'\t‣ Model Files for {new_gene_id} are Loaded')
            else: 
                print(f'\t‣ Error Loading Model Files for {new_gene_id}!!')

        if SEED_FAMILY == 'dfine' and not successful_sub_flag:
            record_dfine_proposal_failure(new_gene_id, 'LLM job submission failed')
        failed_process = not (successful_sub_flag and job_done)
        # Return the new gene ID
        return new_gene_id, failed_process
    
    global GLOBAL_DATA
    global DELAYED_CHECK

    new_gene_id1, failed_process1 = combine_elements(ind1, ind2, llm_model)
    new_gene_id2, failed_process2 = combine_elements(ind2, ind1, llm_model)
    
    if DELAYED_CHECK:
        LINKED_GENES[new_gene_id1] = ind1[0]
        LINKED_GENES[new_gene_id2] = ind2[0]
        ind1[0] = new_gene_id1
        ind2[0] = new_gene_id2
        
        offspring1 = creator.Individual([new_gene_id1])
        offspring2 = creator.Individual([new_gene_id2])
        return offspring1, offspring2

    offspring1 = update_individual(ind1, new_gene_id1, old_gene_id=ind1[0], 
                                   process_success=(not failed_process1), process_type='Mating')
    
    offspring2 = update_individual(ind2, new_gene_id2, old_gene_id=ind2[0], 
                                   process_success=(not failed_process2), process_type='Mating')

    return offspring1, offspring2

def customMutation(individual, llm_model, indpb, temp_min=0.02, temp_max=0.35):
    """
    Custom mutation function that randomly changes the temperature parameter of the individual's task and assigns a new ID.
    Parameters
    ----------
    individual: list 
        The individual to be mutated
    indpb: float 
        The probability of mutating each gene
    Returns
    -------
    individual:
        The mutated individual
    """
    print('customMutation, llm_model:', llm_model)

    global DELAYED_CHECK
    # Check if mutation occurs (based on the mutation probability)
    # if random.random() < indpb: # TODO: connect this to temp
    config = load_yaml()
    old_gene_id = individual[0]
    # Generate a new gene ID
    new_gene_id = generate_random_string(length=24)
    print(f'Mutating: {old_gene_id} and Replacing with: {new_gene_id}')
    # Name of the sh bash file
    file_path = os.path.join(OUTPUT_DIR, str(GENERATION), f'{new_gene_id}.sh')
    temperature = round(random.uniform(temp_min, temp_max), 2)
    successful_sub_flag, job_id, local_output = submit_bash(file_path, 
                                              input_filename_x= f'{VARIANT_DIR}/{MODEL}_{old_gene_id}.py',
                                              output_filename = f'{VARIANT_DIR}/{MODEL}_{new_gene_id}.py',
                                              gpu=config['gpu_selection'],
                                              python_file='src/llm_mutation.py',
                                              job_name="mutation_operation",
                                              top_p=0.1, llm_model=llm_model, temperature=temperature)
    
    # Update the individual with the new gene ID
    # individual[0] = new_gene_id
    # Update the global data with the new task
    GLOBAL_DATA[new_gene_id] = {'sub_flag':successful_sub_flag, 'job_id':job_id, 
                                'status':'subbed file', 'fitness':None, 'start_time':time.time(),
                                'operator_type': 'mutation', 'local_output': local_output}
    if SEED_FAMILY == 'dfine':
        GLOBAL_DATA[new_gene_id]['parent_sources'] = [
            f'{VARIANT_DIR}/{MODEL}_{old_gene_id}.py']
    if SEED_FAMILY == 'yolo26_p2':
        GLOBAL_DATA[new_gene_id]['parent_gene_id'] = old_gene_id
    
    if DELAYED_CHECK:
        LINKED_GENES[new_gene_id] = individual[0]
        GLOBAL_DATA[new_gene_id]['status'] = 'DELAYED_CHECK'
        individual[0] = new_gene_id
        individual = creator.Individual([new_gene_id])
        return individual
    
    if successful_sub_flag:
        print(f'\t‣ Checking for Mutation Job Completion: {job_id} for {new_gene_id}')
        job_done = check4job_completion(job_id, local_output, extension="evolution/")
        if SEED_FAMILY == 'dfine' and not job_done:
            record_dfine_proposal_failure(new_gene_id)
        if job_done:
            print(f'\t‣ Model Files for {new_gene_id} are Loaded')
        else: 
            print(f'\t☠ Error Loading Model Files for {new_gene_id}')
    
    if SEED_FAMILY == 'dfine' and not successful_sub_flag:
        record_dfine_proposal_failure(new_gene_id, 'LLM job submission failed')
    failed_process = not (successful_sub_flag and job_done)

    individual = update_individual(individual, new_gene_id, old_gene_id,
                                   process_success=(not failed_process), process_type='Mutation')
    return individual

def remove_duplicates(population):
    unique_individuals = []
    seen_chromosomes = set()

    for individual in population:
        # Convert chromosome to a tuple since lists are not hashable
        chromosome = tuple(individual)  
        if chromosome not in seen_chromosomes:
            unique_individuals.append(individual)
            seen_chromosomes.add(chromosome)

    return unique_individuals

# --- Checkpoint Functions --- #
_ACTIVE_FINGERPRINT = None

def run_fingerprint():
    """Identify the FRED protocol and inputs that must stay fixed across resume."""
    global _ACTIVE_FINGERPRINT
    if SEED_FAMILY not in ('yolo11', 'yolo26_p2', 'dfine') or "RUN_CHECKPOINT_DIR" not in globals():
        return None

    from sota.FRED_LLM_GE.phase1_detection.protocol_fingerprint import compute_fingerprint
    identity = compute_fingerprint(ROOT_DIR, SEED_FAMILY, RUN_TAG, SEED_NETWORK)
    identity['evolution_settings'] = {
        'num_generations': num_generations,
        'start_population_size': start_population_size,
        'population_size': population_size,
        'offspring_size': offspring_size,
        'num_elites': num_elites,
        'max_refill_attempts': globals().get('FRED_MAX_REFILL_ATTEMPTS'),
        'eval_gpu_constraint': globals().get('EVAL_GPU_CONSTRAINT'),
        'eval_gpu_retry_limit': globals().get('EVAL_GPU_RETRY_LIMIT', 0),
        'preserve_slurm_cuda_visible_devices': globals().get(
            'PRESERVE_SLURM_CUDA_VISIBLE_DEVICES', False),
        'fred_events': globals().get('FRED_EVENTS', ''),
    }
    if SEED_FAMILY in VALIDATED_FITNESS_FAMILIES:
        identity['evolution_settings'].update({
            'llm_model': globals().get('LLM_MODEL'),
            'llm_model_path': globals().get('MODEL_PATH'),
            'llm_max_new_tokens': globals().get('LLM_MAX_NEW_TOKENS'),
            'prompt_group': globals().get('DEFAULT_PROMPT_GROUP'),
            'crossover_template_dir': globals().get('CROSSOVER_TEMPLATE_DIR'),
            'max_gen_attempts': max_gen_attempts,
        })
    if _ACTIVE_FINGERPRINT is not None and identity != _ACTIVE_FINGERPRINT:
        raise RuntimeError('FRED protocol changed during this run; refusing checkpoint update')
    _ACTIVE_FINGERPRINT = identity
    return identity

def save_checkpoint(gen, folder_name="checkpoints", global_path=None, checkpoint_data=None):
    fingerprint = run_fingerprint() if globals().get("RUN_CHECKPOINT_DIR") else None
    from src.fred_session import enabled, save_pair
    if enabled():
        if global_path is None:
            raise ValueError('Bounded sessions require a paired global checkpoint path')
        data = checkpoint_data if checkpoint_data is not None else {'population': population, 'hof': hof}
        state = {'GLOBAL_DATA': GLOBAL_DATA, 'GLOBAL_DATA_HIST': GLOBAL_DATA_HIST,
                 'GLOBAL_DATA_ANCESTRY': GLOBAL_DATA_ANCESTRY}
        if SEED_FAMILY == 'yolo26_p2':
            if 'selection_version' in data or 'coverage_version' in data:
                require_yolo26_selection_version(data, folder_name)
            data = {**data, 'selection_version': YOLO26_SELECTION_VERSION,
                    'coverage_version': YOLO26_COVERAGE_VERSION}
            state['selection_version'] = YOLO26_SELECTION_VERSION
            state['coverage_version'] = YOLO26_COVERAGE_VERSION
        if fingerprint is not None:
            data = dict(data, protocol_fingerprint=fingerprint)
            state['protocol_fingerprint'] = fingerprint
        if load_candidate_policy(globals()) is not None:
            state['EVAL_CACHE'] = EVAL_CACHE
        return save_pair(folder_name, global_path, gen, data, state, SEED_FAMILY)
    atomic_initial_pair = (gen == 0 and global_path is not None and RUN_TAG and
                           SEED_FAMILY in VALIDATED_FITNESS_FAMILIES)
    os.makedirs(folder_name, exist_ok=True)
    filename = os.path.join(folder_name, f'checkpoint_gen_{gen}.pkl')
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            stored_checkpoint = pickle.load(file)
        require_yolo26_selection_version(stored_checkpoint, filename)
        verify_fingerprint(stored_checkpoint, filename, fingerprint)

    if global_path is not None:
        os.makedirs(global_path, exist_ok=True)
        global_file = os.path.join(global_path, f'global_gen_{gen}.pkl')
        if os.path.exists(global_file):
            with open(global_file, "rb") as file:
                try:
                    stored_global_data = pickle.load(file)
                except EOFError:
                    stored_global_data = {}  
            verify_fingerprint(stored_global_data, global_file, fingerprint)
        else:
            stored_global_data = {}

        if stored_global_data:
            require_yolo26_selection_version(stored_global_data, global_file)


        print("ASSIGNING STORED GLOBAL DATA")

        if stored_global_data:
            print("Global data found, updating to file")
            stored_global_data["GLOBAL_DATA"].update(GLOBAL_DATA)
            stored_global_data["GLOBAL_DATA_HIST"].update(GLOBAL_DATA_HIST)
            stored_global_data["GLOBAL_DATA_ANCESTRY"].update(GLOBAL_DATA_ANCESTRY)
            if load_candidate_policy(globals()) is not None:
                stored_global_data.setdefault("EVAL_CACHE", {}).update(EVAL_CACHE)
        else:
            print("No global data found, saving to a new file")
            stored_global_data = {
                "GLOBAL_DATA": GLOBAL_DATA,
                "GLOBAL_DATA_HIST": GLOBAL_DATA_HIST,
                "GLOBAL_DATA_ANCESTRY": GLOBAL_DATA_ANCESTRY,
            }
            if load_candidate_policy(globals()) is not None:
                stored_global_data["EVAL_CACHE"] = EVAL_CACHE
        if SEED_FAMILY == 'yolo26_p2':
            stored_global_data['selection_version'] = YOLO26_SELECTION_VERSION
            stored_global_data['coverage_version'] = YOLO26_COVERAGE_VERSION
        if fingerprint is not None:
            stored_global_data['protocol_fingerprint'] = fingerprint

        if not atomic_initial_pair and SEED_FAMILY in VALIDATED_FITNESS_FAMILIES and RUN_TAG:
            from src.fred_session import atomic_write
            atomic_write(global_file, pickle.dumps(stored_global_data, protocol=pickle.HIGHEST_PROTOCOL))
        elif not atomic_initial_pair:
            with open(global_file, 'wb') as file:
                pickle.dump(stored_global_data, file)
        if not atomic_initial_pair:
            print(f'Global data saved as {global_file}')

    if checkpoint_data is None:
        checkpoint_data = {
            "population": population,
            "hof": hof, 
        }
    if SEED_FAMILY == 'yolo26_p2':
        if 'selection_version' in checkpoint_data or 'coverage_version' in checkpoint_data:
            require_yolo26_selection_version(checkpoint_data, filename)
        checkpoint_data = {**checkpoint_data, 'selection_version': YOLO26_SELECTION_VERSION,
                           'coverage_version': YOLO26_COVERAGE_VERSION}
    if fingerprint is not None:
        checkpoint_data = dict(checkpoint_data, protocol_fingerprint=fingerprint)
    if atomic_initial_pair:
        checkpoint_data = dict(checkpoint_data, _global_state_snapshot=stored_global_data)
    if SEED_FAMILY in VALIDATED_FITNESS_FAMILIES and RUN_TAG:
        from src.fred_session import atomic_write
        atomic_write(filename, pickle.dumps(checkpoint_data, protocol=pickle.HIGHEST_PROTOCOL))
    else:
        with open(filename, 'wb') as file:
            pickle.dump(checkpoint_data, file)
    print(f"Population data saved as {filename}")
    if atomic_initial_pair:
        atomic_write(global_file, pickle.dumps(stored_global_data, protocol=pickle.HIGHEST_PROTOCOL))
        print(f'Global data saved as {global_file}')


def persist_initial_population_checkpoint(active_population=None):
    """Save generation-zero submissions while the controller is still evaluating."""
    if not INITIAL_POPULATION_PENDING or SEED_FAMILY not in VALIDATED_FITNESS_FAMILIES:
        return
    members = list(population)
    if active_population is not None:
        seen = {ind[0] for ind in members}
        members.extend(ind for ind in active_population if ind[0] not in seen)
    save_checkpoint(0, folder_name=RUN_CHECKPOINT_DIR,
                    global_path=RUN_CHECKPOINT_DIR,
                    checkpoint_data={'population': members, 'hof': hof,
                                     'initial_population_pending_evaluation': True})


def persist_initial_creation_checkpoint():
    """Keep submitted initial LLM jobs and their coverage slots resumable."""
    if not INITIAL_CREATION_PENDING or SEED_FAMILY not in VALIDATED_FITNESS_FAMILIES:
        return
    members = [creator.Individual([gene_id]) for gene_id in INITIAL_CREATION_IDS]
    save_checkpoint(0, folder_name=RUN_CHECKPOINT_DIR,
                    global_path=RUN_CHECKPOINT_DIR,
                    checkpoint_data={'population': members, 'hof': hof,
                                     'initial_population_creation_pending': True})

GEN_PATTERNS = {
    "checkpoint": re.compile(r"^checkpoint_gen_(\d+)\.pkl$"),
    "global": re.compile(r"^global_gen_(\d+)\.pkl$"),
}

def extract_generation(filename, kind):
    """Parse generation number from a checkpoint/global filename; return None if it doesn't match."""
    matcher = GEN_PATTERNS[kind].match(os.path.basename(filename))
    return int(matcher.group(1)) if matcher else None

def load_checkpoint(folder_name="checkpoints", checkpoint_file=None, global_path="checkpoints", global_file=None):
    fingerprint = run_fingerprint() if globals().get("RUN_CHECKPOINT_DIR") else None
    from src.fred_session import enabled, eval_submission, latest_pair
    if enabled():
        if checkpoint_file is not None or global_file is not None:
            raise ValueError('Bounded sessions resume the newest committed checkpoint pair')
        data, start_gen, state = latest_pair(folder_name, global_path, SEED_FAMILY)
        if data is not None:
            require_yolo26_selection_version(data, folder_name)
            require_yolo26_selection_version(state, global_path)
            verify_fingerprint(data, folder_name, fingerprint)
            verify_fingerprint(state, global_path, fingerprint)
            if SEED_FAMILY == 'dfine':
                for individual in data['population']:
                    gene_id = individual[0]
                    record = state['GLOBAL_DATA'].get(gene_id)
                    if not record or record['status'] not in ('subbed file', 'running eval'):
                        continue
                    submission = eval_submission(folder_name, gene_id)
                    if submission and submission['state'] == 'submitted':
                        record.update(status='running eval', results_job=submission['job_id'],
                                      eval_start_time=submission['at'], local_output=None)
                    elif submission and submission['state'] == 'cancelled':
                        record.update(status='subbed file', results_job=None, local_output=None)
                    elif submission:
                        record.update(status='FAILED: INFRASTRUCTURE', fitness=INVALID_FITNESS_MAX,
                                      failure_reason='Evaluation submission was interrupted or failed; '
                                                     'job identity could not be confirmed')
        return data, start_gen, state
    if fingerprint is not None and os.path.exists(folder_name) != os.path.exists(global_path):
        raise RuntimeError('FRED checkpoint is incomplete: population and global paths must both exist')
    if not os.path.exists(folder_name) or not os.path.exists(global_path):
        print("Path does not exist, returning none for checkpoints")
        return None, None, None

    population_data = None
    start_gen = 0
    global_data = {}

    if checkpoint_file is None:
        checkpoint_candidates = [f for f in os.listdir(folder_name) if extract_generation(f, "checkpoint") is not None]
        checkpoint_files = sorted(checkpoint_candidates, key=lambda f: extract_generation(f, "checkpoint"), reverse=True)
        checkpoint_file = checkpoint_files[0] if checkpoint_files else None
    if checkpoint_file:
        filepath = os.path.join(folder_name, checkpoint_file)
        with open(filepath, 'rb') as file:
            population_data = pickle.load(file)
        require_yolo26_selection_version(population_data, filepath)
        verify_fingerprint(population_data, filepath, fingerprint)
        print(f"Loaded population data from {filepath}")
        start_gen = extract_generation(checkpoint_file, "checkpoint") + 1
        if (SEED_FAMILY in VALIDATED_FITNESS_FAMILIES and start_gen == 1
                and (population_data.get('initial_population_pending_evaluation') or
                     population_data.get('initial_population_creation_pending'))):
            start_gen = 0

    embedded_state = (population_data.get('_global_state_snapshot')
                      if population_data and SEED_FAMILY in VALIDATED_FITNESS_FAMILIES else None)
    if embedded_state is not None:
        global_data = embedded_state
        require_yolo26_selection_version(global_data, checkpoint_file)
        verify_fingerprint(global_data, checkpoint_file, fingerprint)
        print(f'Loaded matching generation-0 global state from {checkpoint_file}')
    else:
        if global_file is None:
            global_candidates = [f for f in os.listdir(global_path) if extract_generation(f, "global") is not None]
            global_files = sorted(global_candidates, key=lambda f: extract_generation(f, "global"), reverse=True)
            global_file = global_files[0] if global_files else None
        if global_file:
            filepath = os.path.join(global_path, global_file)
            with open(filepath, 'rb') as file:
                global_data = pickle.load(file)
            require_yolo26_selection_version(global_data, filepath)
            verify_fingerprint(global_data, filepath, fingerprint)
            print(f"Loaded global data from {filepath}")

    if fingerprint is not None and bool(population_data) != bool(global_data):
        raise RuntimeError('FRED checkpoint is incomplete: population and global data must both exist')
    if fingerprint is not None and embedded_state is None and checkpoint_file and global_file:
        if extract_generation(checkpoint_file, 'checkpoint') != extract_generation(global_file, 'global'):
            raise RuntimeError('FRED checkpoint generations do not match')

    return population_data, start_gen, global_data

def select_elites(pop, k):
    """Select YOLO26 by mAP50:95; keep the legacy SPEA2 policy elsewhere.

    For legacy families, KEEP_BEST_ELITE reserves the best by the first
    objective because SPEA2's density step can otherwise favor model size.
    """
    if SEED_FAMILY == 'yolo26_p2':
        return select_yolo26_by_accuracy(pop, k)
    if not globals().get("KEEP_BEST_ELITE", False) or k < 1 or not pop:
        return tools.selSPEA2(pop, k)
    best = tools.selBest(pop, 1)[0]
    rest = [ind for ind in pop if ind[0] != best[0]]
    return [best] + tools.selSPEA2(rest, min(k - 1, len(rest)))


def select_yolo26_by_accuracy(pop, k):
    """Rank by mAP50:95; use fewer parameters only when accuracy ties."""
    return sorted(pop, key=lambda ind: (ind.fitness.values[1], -ind.fitness.values[2]),
                  reverse=True)[:k]


def true_nsga2(pop, k):
    pop = tools.selNSGA2(pop, len(pop)) # 10 diff
    # selTournamentDCD needs a multiple of 4, so round up and trim back to k
    # instead of rounding down (3 survivors used to select 0 offspring).
    k4 = -(-k // 4) * 4
    pop = k4 * pop
    new_pop = tools.selTournamentDCD(pop, k4) # mults of 4
    return new_pop[:k]


def select_parents(pop, k):
    if SEED_FAMILY == 'yolo26_p2':
        return select_yolo26_by_accuracy(pop, k)
    return true_nsga2(pop, k)

def create_population(n, llm_model, coverage_panel=False):
    if SEED_FAMILY == 'yolo26_p2' and coverage_panel:
        from src.fred_yolo26_proposals import initial_target_for_slot
        return [toolbox.individual(llm_model=llm_model,
                                   coverage_target=initial_target_for_slot(slot))
                for slot in range(n)]
    individual_func = partial(toolbox.individual, llm_model=llm_model)
    return tools.initRepeat(list, individual_func, n)


def distinct_built_population(individuals, policy):
    """Keep one successful individual per effective built graph, in input order."""
    unique = []
    seen = set()
    for individual in individuals:
        entry = GLOBAL_DATA.get(individual[0], {})
        if entry.get('fitness') in (None, INVALID_FITNESS_MAX) or entry.get('failure_reason'):
            continue
        graph_hash = entry.get('architecture_hash')
        if graph_hash is None:
            path = f'{VARIANT_DIR}/{MODEL}_{individual[0]}.py'
            graph_hash, params = policy.built_architecture(path)
            entry.update(architecture_hash=graph_hash, architecture_params=params)
        if graph_hash not in seen:
            seen.add(graph_hash)
            unique.append(individual)
    return unique


def write_run_status(status):
    run_root = globals().get('RUN_ROOT')
    if run_root:
        os.makedirs(run_root, exist_ok=True)
        with open(os.path.join(run_root, 'status.json'), 'w') as stream:
            json.dump(status, stream, indent=2)


def write_depleted_status(generation, found, target, attempts):
    message = (f'depleted: generation {generation} has {found}/{target} distinct successful '
               f'built graphs after {attempts} refill attempts')
    write_run_status({'status': 'depleted', 'generation': generation, 'distinct_graphs': found,
                      'target': target, 'refill_attempts': attempts})
    raise RuntimeError(message)


def refill_initial_population(individuals, target, llm_model):
    """Start YOLO11 generation 0 with distinct built graphs or stop clearly."""
    global GEN_COUNT
    policy = load_candidate_policy(globals())
    if policy is None:
        return individuals
    GEN_COUNT = 0  # Refill mutations need lineage; EoT remains disabled at generation 0.
    generation = 0
    population = distinct_built_population(individuals, policy)
    limit = globals().get('FRED_MAX_REFILL_ATTEMPTS', 3 * target)
    attempts = 0
    all_fresh = []
    while len(population) < target and attempts < limit:
        if not population:
            log_failure_counts(generation, [*individuals, *all_fresh])
            write_depleted_status(generation, 0, target, attempts)
        count = min(target - len(population), limit - attempts)
        fresh = []
        for _ in range(count):
            child = toolbox.clone(random.choice(population))
            fresh.append(toolbox.mutate(individual=child, llm_model=llm_model))
        attempts += count
        delayed_mutate_check(fresh)
        for child in fresh:
            child.fitness.values = PLACEHOLDER_FITNESS
        check_and_update_fitness(fresh)
        all_fresh.extend(fresh)
        population = distinct_built_population([*population, *fresh], policy)
        print(f'Generation {generation} distinct graphs: {len(population)}/{target}; '
              f'refill attempts: {attempts}/{limit}', flush=True)
    if len(population) < target:
        log_failure_counts(generation, [*individuals, *all_fresh])
        write_depleted_status(generation, len(population), target, attempts)
    log_failure_counts(generation, [*individuals, *all_fresh])
    return population


def distinct_valid_detection_population(candidates):
    """Keep one successful candidate per checked architecture identity."""
    if SEED_FAMILY == 'yolo26_p2':
        from sota.FRED_LLM_GE.phase1_detection.adapters.ultralytics_yolo26_p2 import (
            UltralyticsYolo26P2Adapter,
        )
        architecture_hash = UltralyticsYolo26P2Adapter.architecture_hash
    elif SEED_FAMILY == 'dfine':
        from src.fred_dfine_proposals import architecture_hash
    else:
        raise ValueError(f'No custom architecture identity for {SEED_FAMILY}')
    unique, seen = [], {}
    for individual in candidates:
        record = GLOBAL_DATA[individual[0]]
        if record.get('fitness') in (None, INVALID_FITNESS_MAX) or record.get('failure_reason'):
            continue
        digest = record.get('architecture_sha256')
        if digest is None:
            digest = architecture_hash(f'{VARIANT_DIR}/{MODEL}_{individual[0]}.py')
            record['architecture_sha256'] = digest
        if digest in seen:
            if seen[digest] != individual[0]:
                record['population_exclusion_reason'] = (
                    f'duplicate_architecture: matches {seen[digest]} ({digest})')
                print(f'\t☠ Excluding {SEED_FAMILY} candidate {individual[0]}: '
                      f"{record['population_exclusion_reason']}", flush=True)
            continue
        record.pop('population_exclusion_reason', None)
        unique.append(individual)
        seen[digest] = individual[0]
    return unique


def refill_detection_initial_population(individuals, target, llm_model):
    """Require a complete, distinct custom-detector generation zero."""
    global GEN_COUNT
    if SEED_FAMILY == 'yolo26_p2':
        from src.fred_yolo26_proposals import INITIAL_COVERAGE_TARGETS
        for individual in individuals[1:1 + len(INITIAL_COVERAGE_TARGETS)]:
            record = GLOBAL_DATA[individual[0]]
            if record.get('fitness') in (None, INVALID_FITNESS_MAX) or record.get('failure_reason'):
                write_run_status({'status': 'coverage_failed', 'generation': 0,
                                  'target': record.get('coverage_target'),
                                  'failure_reason': record.get('failure_reason')})
                raise RuntimeError(f"YOLO26 required {record.get('coverage_target')} candidate "
                                   f"{individual[0]} failed evaluation; generation 0 is incomplete")
    elif SEED_FAMILY != 'dfine':
        raise ValueError(f'No custom initial refill for {SEED_FAMILY}')

    population = distinct_valid_detection_population(individuals)
    limit = globals().get('FRED_MAX_REFILL_ATTEMPTS', 3 * target)
    attempts = 0
    all_fresh = []
    while len(population) < target and attempts < limit:
        count = min(target - len(population), limit - attempts)
        current_gen = GEN_COUNT
        GEN_COUNT = -1
        try:
            fresh = toolbox.population(n=count, llm_model=llm_model)
            delayed_creation_check(fresh)
        finally:
            GEN_COUNT = current_gen
        attempts += count
        for child in fresh:
            child.fitness.values = PLACEHOLDER_FITNESS
        check_and_update_fitness(fresh)
        all_fresh.extend(fresh)
        population = distinct_valid_detection_population([*population, *fresh])
        print(f'Generation 0 distinct {SEED_FAMILY} architectures: {len(population)}/{target}; '
              f'refill attempts: {attempts}/{limit}', flush=True)
    log_failure_counts(0, [*individuals, *all_fresh])
    if len(population) < target:
        write_run_status({'status': 'depleted', 'generation': 0,
                          'distinct_architectures': len(population), 'target': target,
                          'refill_attempts': attempts})
        raise RuntimeError(f'{SEED_FAMILY} generation 0 has {len(population)}/{target} distinct '
                           f'successful architectures after {attempts} refill attempts')
    return population


def refill_detection_generation_population(individuals, target, llm_model, generation):
    """Replace failed or duplicate offspring while keeping valid survivors."""
    population = distinct_valid_detection_population(individuals)
    limit = globals().get('FRED_MAX_REFILL_ATTEMPTS', 3 * target)
    attempts = 0
    all_fresh = []
    while len(population) < target and attempts < limit:
        if not population:
            break
        count = min(target - len(population), limit - attempts)
        fresh = [toolbox.mutate(individual=toolbox.clone(random.choice(population)),
                                llm_model=llm_model) for _ in range(count)]
        attempts += count
        delayed_mutate_check(fresh)
        for child in fresh:
            child.fitness.values = PLACEHOLDER_FITNESS
        check_and_update_fitness(fresh)
        all_fresh.extend(fresh)
        population = distinct_valid_detection_population([*population, *fresh])
        print(f'Generation {generation} distinct {SEED_FAMILY} architectures: '
              f'{len(population)}/{target}; refill attempts: {attempts}/{limit}', flush=True)
    log_failure_counts(generation, [*individuals, *all_fresh])
    if len(population) < target:
        write_run_status({'status': 'depleted', 'generation': generation,
                          'distinct_architectures': len(population), 'target': target,
                          'refill_attempts': attempts})
        raise RuntimeError(f'{SEED_FAMILY} generation {generation} has '
                           f'{len(population)}/{target} distinct successful architectures '
                           f'after {attempts} refill attempts')
    return population


def add_stock_seed_to_initial_population(population, allow_existing=False):
    """Give a FRED stock seed a normal candidate ID and evaluation job.

    The copy is byte-for-byte, so Python seeds and YAML seeds use the same
    candidate_runner path as their LLM-created descendants. The standalone
    three-seed baseline remains independent of this single-seed GE individual.
    """
    if "SEED_FAMILY" not in globals():
        return population
    gene_id = "seed"
    os.makedirs(VARIANT_DIR, exist_ok=True)
    target = os.path.join(VARIANT_DIR, f"{MODEL}_{gene_id}.py")
    if gene_id in GLOBAL_DATA or (os.path.exists(target) and not allow_existing):
        raise RuntimeError(f"Stock seed already exists in a fresh run: {target}; use a fresh run tag")
    if os.path.exists(target):
        with open(target, 'rb') as current, open(SEED_NETWORK, 'rb') as source:
            if current.read() != source.read():
                raise RuntimeError(f'Stock seed file changed during resume: {target}')
    else:
        shutil.copyfile(SEED_NETWORK, target)
    GLOBAL_DATA[gene_id] = {"sub_flag": True, "job_id": None,
                            "status": "subbed file", "fitness": None,
                            "start_time": time.time(), "origin": "stock_seed"}
    GLOBAL_DATA_ANCESTRY[gene_id] = {"GENES": [gene_id], "MUTATE_TYPE": ["STOCK_SEED"]}
    return [creator.Individual([gene_id]), *population]


def verify_yolo26_initial_population(population):
    """Record complete, distinct P2–P5 coverage before any training jobs."""
    if SEED_FAMILY != 'yolo26_p2':
        return
    from src.fred_yolo26_proposals import INITIAL_COVERAGE_TARGETS, verify_initial_coverage
    targets = [GLOBAL_DATA[ind[0]]['coverage_target'] for ind in population]
    expected = list(INITIAL_COVERAGE_TARGETS) + [None] * (len(population) - len(INITIAL_COVERAGE_TARGETS))
    if targets != expected:
        raise ValueError(f'Initial YOLO26 target order must be {expected}, got {targets}')
    entries = [(GLOBAL_DATA[ind[0]]['coverage_target'],
                os.path.join(VARIANT_DIR, f'{MODEL}_{ind[0]}.py'))
               for ind in population if GLOBAL_DATA[ind[0]]['coverage_target'] is not None]
    report = verify_initial_coverage(entries)
    report['open_candidates'] = [ind[0] for ind in population[len(INITIAL_COVERAGE_TARGETS):]]
    report_path = os.path.join(OUTPUT_DIR, 'initial_yolo26_coverage.json')
    if os.path.exists(report_path):
        with open(report_path) as handle:
            if json.load(handle) != report:
                raise RuntimeError(f'Initial YOLO26 coverage changed during resume: {report_path}')
    else:
        with open(report_path, 'x') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
    print(f'Initial YOLO26 coverage verified: {report_path}', flush=True)


def finish_initial_creation(individuals, llm_model, resume=False):
    """Complete submitted generation-0 LLM jobs before starting seed evaluation."""
    global INITIAL_CREATION_PENDING
    if resume:
        target = start_population_size - 1
        if len(individuals) > target:
            raise RuntimeError(f'Generation-0 creation checkpoint has {len(individuals)} '
                               f'LLM candidates; expected at most {target}')
        for slot in range(len(individuals), target):
            if SEED_FAMILY == 'yolo26_p2':
                from src.fred_yolo26_proposals import initial_target_for_slot
                candidate = toolbox.individual(llm_model=llm_model,
                                               coverage_target=initial_target_for_slot(slot))
            else:
                candidate = toolbox.individual(llm_model=llm_model)
            individuals.append(candidate)
    INITIAL_CREATION_PENDING = False
    delayed_creation_check(individuals)
    verify_yolo26_initial_population(individuals)
    return add_stock_seed_to_initial_population(individuals, allow_existing=resume)

# Define the problem
creator.create("FitnessMulti", base.Fitness, weights=FITNESS_WEIGHTS)  # Adjust weights as needed
creator.create("Individual", list, fitness=creator.FitnessMulti, file_id=None)

# Initialize the toolbox
toolbox = base.Toolbox()
toolbox.register("individual", create_individual, creator.Individual)
toolbox.register("population", create_population)
toolbox.register("evaluate", evalModel)
toolbox.register("mate", customCrossover)
toolbox.register("mutate", customMutation, indpb=0.2)
toolbox.register("select", select_parents)

# TODO: start using percent diff of train acc vs val test acc as an over fitt metric 

# 40398682
GEN_COUNT = -1
TOP_N_GENES = None
LINKED_GENES = {}
GLOBAL_DATA = {}
GLOBAL_DATA_HIST = {}
EVAL_CACHE = {}
EVAL_PENDING = {}
GLOBAL_DATA_ANCESTRY = {}
GLOBAL_DATA_ANCESTRY[MODEL] = {'GENES':[MODEL], 'MUTATE_TYPE':["CREATED"]}
CHECKPOINT_FOLDER = None

# Main Evolution Loop
if __name__ == "__main__":
    # make output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set Cluter Configurations
    parser = argparse.ArgumentParser(description='Run Generation')
    # Add arguments
    parser.add_argument('--checkpoints', type=str, help='Save Dir')
    parser.add_argument('--llm_model', type=str, help='Which LLM to use', default=DEFAULT_LLM_MODEL)
    parser.add_argument('--global_path', type=str, help='Path to global variables', default=ROOT_DIR)
    parser.add_argument('--prompt_group', type=str, help='Prompt group or glob (relative to templates/)', default=None)
    # Parse the arguments
    args = parser.parse_args()
    llm_model = args.llm_model
    CHECKPOINT_FOLDER = args.checkpoints

    # run.sh supplies legacy checkpoint names. An explicit FRED run tag keeps
    # both checkpoint files in the same experiment-specific directory.
    if "RUN_CHECKPOINT_DIR" in globals():
        if not RUN_CHECKPOINT_DIR:
            parser.error(
                "set a fresh FRED_RUN_TAG, e.g. yolo11_20261017_eot0_trial1"
            )
        args.checkpoints = RUN_CHECKPOINT_DIR
        args.global_path = RUN_CHECKPOINT_DIR
        print(f"FRED run tag: {RUN_TAG}; checkpoints: {RUN_CHECKPOINT_DIR}", flush=True)
    CHECKPOINT_FOLDER = args.checkpoints

    PROMPT_GLOB = resolve_prompt_glob(args.prompt_group)

    print(DNA_TXT)
    print("ISLAND llm_model: ", llm_model)
    print(f"Prompt glob: {PROMPT_GLOB}")
    if load_candidate_policy(globals()) is not None or (RUN_TAG and SEED_FAMILY in VALIDATED_FITNESS_FAMILIES):
        print(f"FRED generation limit: {num_generations - 1} evolutionary generations; "
              f"LLM server job: {os.getenv('LLMGE_SERVER_JOB_ID', '<not supplied>')}", flush=True)

    if AVAILABLE_LLM_MODELS and (not llm_model or llm_model not in AVAILABLE_LLM_MODELS):
        print("Error in Island Generation: No LLM specified. Exiting script")
        exit(1)

    population_data, start_gen, global_data = load_checkpoint(folder_name=args.checkpoints, global_path=args.global_path)
    initial_pending = bool(population_data and
                           population_data.get('initial_population_pending_evaluation'))
    creation_pending = bool(population_data and
                            population_data.get('initial_population_creation_pending'))
    if population_data:
        box_print("CHECKPOINT LOADED")
        GLOBAL_DATA = global_data["GLOBAL_DATA"]
        GLOBAL_DATA_HIST = global_data["GLOBAL_DATA_HIST"]
        GLOBAL_DATA_ANCESTRY = global_data["GLOBAL_DATA_ANCESTRY"]
        if load_candidate_policy(globals()) is not None:
            from sota.FRED_LLM_GE.phase1_detection.ancestry_recovery import recover_population_ancestry
            GLOBAL_DATA_ANCESTRY, repaired = recover_population_ancestry(
                population_data["population"], GLOBAL_DATA_ANCESTRY, ROOT_DIR, OUTPUT_DIR,
                VARIANT_DIR, MODEL, PROMPT_GLOB, CONSTANT_RULES_PATH)
            if repaired:
                print(f"Recovered ancestry for generation-0 refill candidates: {repaired}",
                      flush=True)
        EVAL_CACHE = global_data.get("EVAL_CACHE", {})
        EVAL_PENDING = {}
        population = population_data["population"]
        hof = population_data["hof"]
        if SEED_FAMILY in VALIDATED_FITNESS_FAMILIES:
            valid_hof = valid_detection_population(list(hof))
            if len(valid_hof) != len(hof):
                hof.clear()
                hof.update(valid_hof)
    else:
        # Create an initial population
        start_gen = 1
        box_print("CREATING POPULATION FROM SEED CODE")
        initial_mutations = start_population_size - int("SEED_FAMILY" in globals())
        if initial_mutations < 0:
            raise ValueError("start_population_size must include a slot for the stock seed")
        if SEED_FAMILY == 'yolo26_p2' and initial_mutations < 4:
            raise ValueError("YOLO26 initial population needs four coverage candidates and a stock seed")
        hof = tools.ParetoFront()
        from src.fred_session import enabled
        INITIAL_CREATION_PENDING = (SEED_FAMILY in VALIDATED_FITNESS_FAMILIES and
                                    not enabled())
        INITIAL_CREATION_IDS = []
        if INITIAL_CREATION_PENDING:
            persist_initial_creation_checkpoint()
        if SEED_FAMILY == 'yolo26_p2':
            population = toolbox.population(n=initial_mutations, llm_model=llm_model,
                                            coverage_panel=True)
        else:
            population = toolbox.population(n=initial_mutations, llm_model=llm_model)
        box_print("Batch Checking Created Genes", print_bbox_len=60, new_line_end=False)
        population = finish_initial_creation(population, llm_model)

        # Preserve the initial population and submitted worker identities before
        # long evaluations. Generation zero is setup, not a completed search generation.
        if enabled() or SEED_FAMILY in VALIDATED_FITNESS_FAMILIES:
            INITIAL_POPULATION_PENDING = SEED_FAMILY in VALIDATED_FITNESS_FAMILIES and not enabled()
            if INITIAL_POPULATION_PENDING:
                persist_initial_population_checkpoint()
            else:
                save_checkpoint(0, folder_name=args.checkpoints, global_path=args.global_path)

    if creation_pending:
        INITIAL_CREATION_PENDING = True
        INITIAL_CREATION_IDS = [individual[0] for individual in population]
        population = finish_initial_creation(population, llm_model, resume=True)
        INITIAL_POPULATION_PENDING = True
        persist_initial_population_checkpoint()
        initial_pending = True

    if initial_pending:
        INITIAL_POPULATION_PENDING = True

    # Evaluate the entire population
    for ind in population:
        ind.fitness.values = PLACEHOLDER_FITNESS
        
    check_and_update_fitness(population)
    if not population_data or initial_pending:
        log_failure_counts(0, population)
        if "SEED_FAMILY" in globals():
            if GLOBAL_DATA["seed"]["fitness"] == INVALID_FITNESS_MAX:
                raise RuntimeError("Stock seed failed evaluation; generation 0 was not checkpointed")
            if load_candidate_policy(globals()) is not None:
                population = refill_initial_population(
                    population, start_population_size, llm_model)
            elif SEED_FAMILY in VALIDATED_FITNESS_FAMILIES:
                population = refill_detection_initial_population(
                    population, start_population_size, llm_model)
            hof.update(population)
            INITIAL_POPULATION_PENDING = False
            save_checkpoint(0, folder_name=args.checkpoints, global_path=args.global_path,
                            checkpoint_data={'population': population, 'hof': hof,
                                             'initial_population_pending_evaluation': False})
            start_gen = 1
    # Evolution
    paused_for_server = False
    for gen in range(start_gen, num_generations if migration_gen == 0 else ((start_gen + migration_gen - 1) // migration_gen) * migration_gen + 1):
        if load_candidate_policy(globals()) is not None or (RUN_TAG and SEED_FAMILY in VALIDATED_FITNESS_FAMILIES):
            try:
                check_server_window()
            except ServerWindowClosed as exc:
                print(f'paused_for_server_refresh: {exc}; resume from the previous checkpoint',
                      flush=True)
                paused_for_server = True
                break
        GEN_COUNT = gen
        if SEED_FAMILY not in VALIDATED_FITNESS_FAMILIES:
            TOP_N_GENES = tools.selSPEA2(population, NUM_EOT_ELITES)
        box_print(f"STARTING GENERATION: {gen}", new_line_end=False)
        print_population(population, GLOBAL_DATA)
        box_print(f"Invalid Removal", print_bbox_len=60, new_line_end=False)
        # Remove failed candidates before parent or EoT selection.
        if SEED_FAMILY in VALIDATED_FITNESS_FAMILIES:
            population = valid_detection_population(population)
            population = refill_detection_population(population, llm_model)
            TOP_N_GENES = (select_yolo26_by_accuracy(population, NUM_EOT_ELITES)
                           if SEED_FAMILY == 'yolo26_p2'
                           else tools.selSPEA2(population, NUM_EOT_ELITES))
        else:
            population = [ind for ind in population if ind.fitness.values != INVALID_FITNESS_MAX]

        '''
        IF POPULATION IS LESS THAN NUM_ELITE INDIVIDUALS, GENERATE MORE FROM SCRATCH
         * todo: clean up this code/consolidate with previous code *
        '''

        box_print("CURRENT POPULATION SIZE:", len(population))
        while SEED_FAMILY not in VALIDATED_FITNESS_FAMILIES and len(population) < num_elites:
            print("MINIMUM NUMBER OF IND NOT ACHIEVED, TRYING AGAIN")
            if "SEED_FAMILY" in globals():
                fresh = toolbox.population(
                    n=max(start_population_size - len(population), num_elites - len(population)),
                    llm_model=llm_model)
                delayed_creation_check(fresh)
                for ind in fresh:
                    ind.fitness.values = PLACEHOLDER_FITNESS
                check_and_update_fitness(fresh)
                log_failure_counts(gen, fresh)
                population.extend(ind for ind in fresh if ind.fitness.values != INVALID_FITNESS_MAX)
                continue
            GEN_COUNT = -1
            TOP_N_GENES = None
            LINKED_GENES = {}
            GLOBAL_DATA = {}
            GLOBAL_DATA_HIST = {}
            GLOBAL_DATA_ANCESTRY = {}
            start_gen = 0
            population = toolbox.population(n=start_population_size, llm_model=llm_model)
            delayed_creation_check(population)
            for ind in population:
                ind.fitness.values = PLACEHOLDER_FITNESS
            check_and_update_fitness(population)
            log_failure_counts(gen, population)
            population = [ind for ind in population if ind.fitness.values != INVALID_FITNESS_MAX]
            box_print("CURRENT POPULATION SIZE:", len(population))

        print_population(population, GLOBAL_DATA)
        # Select the next generation's parents
        box_print(f"Selection", print_bbox_len=60, new_line_end=False)
        # These bypass the mutation and cross-over so we dont lose them

        elites = select_elites(population, num_elites)

        # Select the next generation's parents
        offspring_count = min(len(population), globals().get("offspring_size", population_size))
        print(f"Selecting {offspring_count} offspring")
        offspring = toolbox.select(population, offspring_count)
        
        print_population(offspring, GLOBAL_DATA)
        
        print([len(GLOBAL_DATA_HIST), len(GLOBAL_DATA), len(population), len(offspring)])
        
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        GLOBAL_DATA_HIST.update(GLOBAL_DATA.copy())

        # Apply crossover on the offspring
        box_print("Mating", print_bbox_len=60, new_line_end=False)
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_probability:
                child1, child2 = toolbox.mate(child1, child2, llm_model=llm_model)
                del child1.fitness.values
                del child2.fitness.values 
                
        box_print("Batch Checking Mated Genes", print_bbox_len=60, new_line_end=False)       
        offspring = delayed_mate_check(offspring)
        print_population(offspring, GLOBAL_DATA)
        
        # Apply mutation on the offspring
        box_print("Mutating", print_bbox_len=60, new_line_end=False)
        mutate_selected_offspring(offspring, llm_model)
                
        box_print(f"GLOBAL_DATA_ANCESTRY", new_line_end=False)
        print_ancestry(GLOBAL_DATA_ANCESTRY)
                
        box_print("Batch Checking Mutated Genes", print_bbox_len=60, new_line_end=False)
        offspring = delayed_mutate_check(offspring)
        print_population(offspring, GLOBAL_DATA)
        
        # Add elites back to offspring. Usually before the mute and cross but in this case we save them
        offspring.extend(elites)
        # After merging the offspring and the elites
        offspring = remove_duplicates(offspring)
        elites_keys = [k[0] for k in elites]
        # Bring back the elite history
        for k in elites_keys:
            if k in GLOBAL_DATA_HIST.keys():
                GLOBAL_DATA[k] = GLOBAL_DATA_HIST[k]
            """
            GLOBAL_DATA should have the job information and fitness values
            When it hits the below in check_and_update_fitness it will load the results from the dict 
                if GLOBAL_DATA[gene_id]['status'] == "completed":
            """
           
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)

        for ind in offspring:
            # assign placeholder to all so I can check them all at once
            ind.fitness.values = PLACEHOLDER_FITNESS 

        GLOBAL_DATA_HIST.update(GLOBAL_DATA.copy())
        check_and_update_fitness(offspring)
        log_failure_counts(gen, offspring)
        GLOBAL_DATA_HIST.update(GLOBAL_DATA.copy())
        # Replace the old population with the offspring
        population[:] = (valid_detection_population(offspring)
                         if SEED_FAMILY in VALIDATED_FITNESS_FAMILIES else offspring)
        if SEED_FAMILY in VALIDATED_FITNESS_FAMILIES:
            population[:] = refill_detection_generation_population(
                population, population_size, llm_model, gen)
        # Gather all the fitnesses in one list and print the stats
        if SEED_FAMILY not in VALIDATED_FITNESS_FAMILIES or population:
            print_scores(population, FITNESS_WEIGHTS)
            hof.update(population)
        else:
            print(f'Generation {gen} has no valid {SEED_FAMILY} candidates; '
                  'refill on resume/next generation', flush=True)
        save_checkpoint(gen, folder_name=args.checkpoints, global_path=args.global_path)
        write_run_status({'status': 'running', 'last_completed_generation': gen})
        LINKED_GENES = {}
        # mutate x prompts
        # mutate_prompts()

        if SEED_FAMILY not in VALIDATED_FITNESS_FAMILIES or population:
            best_ind = (select_yolo26_by_accuracy(population, 1)[0]
                        if SEED_FAMILY == 'yolo26_p2' else tools.selBest(population, 1)[0])
            print(f"Finished Generation {gen}")
            print(f"Best Individual: {best_ind}")
            print(f"Best Fitness: {best_ind.fitness.values}")
        
    if not paused_for_server:
        write_run_status({'status': 'complete', 'last_completed_generation': num_generations - 1})
    print("-- Paused for server refresh --" if paused_for_server else "-- End of Era --")
