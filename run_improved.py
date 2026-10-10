import os
import copy
import glob
import time
import json
import string
import random
import pickle
import argparse
import subprocess
import yaml
import numpy as np
import re
import urllib.request
import urllib.error
from deap import base, creator, tools
from deap.tools import HallOfFame, ParetoFront
from functools import partial
from src.utils.print_utils import print_population, print_scores, box_print, print_job_info
from src.llm_utils import split_file, retrieve_base_code
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
    if LLM_SERVER_READY:
        return True

    env_timeout = os.getenv("LLM_SERVER_READY_TIMEOUT")
    env_interval = os.getenv("LLM_SERVER_READY_CHECK_INTERVAL")
    if env_timeout:
        timeout = int(env_timeout)
    if env_interval:
        check_interval = int(env_interval)

    start_time = time.time()
    hostname = None
    last_error = None

    while time.time() - start_time <= timeout:
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
    if last_error is not None:
        print(f"\t‣ Last readiness error: {last_error}", flush=True)
    return False

def submit_bash(file_path, **kwargs):
    """ This should be general for subbing anything and returning:
        successful_sub_flag 
        job_id
    """
    if not wait_for_llm_server():
        return False, None, None

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
    
def create_individual(container, llm_model, temp_min=0.05, temp_max=0.4):
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
                                            top_p=0.1, temperature=temperature)
    GLOBAL_DATA[gene_id] = {'sub_flag':successful_sub_flag, 'job_id':job_id, 
                            'status':'subbed file', 'fitness':None, 'start_time':time.time()}
    GLOBAL_DATA_ANCESTRY[gene_id] = {'GENES':[gene_id], 'MUTATE_TYPE':["CREATED"]}
    
    individual = container([gene_id])  # Assign a file ID
    
    if DELAYED_CHECK:
        GLOBAL_DATA[gene_id]['status'] = 'DELAYED_CHECK'
        individual = creator.Individual([gene_id])
        return individual
    
    if successful_sub_flag:
        if job_id is not None:
            print(f'Checking for Job Completion: {job_id} for {gene_id}', flush=True)
        else:
            print(f'Checking completion for {gene_id}', flush=True)
        job_done = check4job_completion(job_id=job_id, local_output=local_output, extension="evolution/")
        print(f'Model Files for {gene_id} are Loaded') if job_done else print(f'Error Loading Model Files for {gene_id}', flush=True)
    return individual

def submit_run(gene_id):
    def write_bash_script_py(gene_id, train_file=f'{TRAIN_FILE}'):
        model_file_override = RUNLINE_TMP.format(MODEL, gene_id) 
        python_runline = EVAL_RUNLINE.format(train_file, model_file_override, VARIANT_DIR=VARIANT_DIR)
        config = load_yaml()
        bash_script_content = fill_template_slots(config['python_bash_script'], python_runline)
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
        result = subprocess.run([RUN_COMMAND, file_path], capture_output=True, text=True)
        if LOCAL:
            local_output = result.stdout.strip() + '\n' + result.stderr.strip()
            print("\t‣ Output:", local_output, flush=True)
            job_id = None
            successful_sub_flag = True
        elif result.returncode == 0:
            print("\t‣ Script Submitted Successfully.\n\t‣ Output:", result.stdout.strip())
            successful_sub_flag = True
            job_id = result.stdout.split('job ')[-1].strip()
        else:
            print("\t‣ Failed to Submit script.\n\t‣ Error:", result.stderr.strip())
            successful_sub_flag = False
            job_id = None
        return successful_sub_flag, job_id, local_output
    
    out_dir = os.path.join(OUTPUT_DIR, str(GENERATION))
    file_path = os.path.join(out_dir, f'{gene_id}_model.sh')
    successful_sub_flag, job_id, local_output = submit_bash_py(file_path, gene_id)
    GLOBAL_DATA[gene_id]['status'] = 'running eval'
    GLOBAL_DATA[gene_id]['results_job'] = job_id
    GLOBAL_DATA[gene_id]['eval_start_time'] = time.time()
    GLOBAL_DATA[gene_id]['local_output'] = local_output
    print(f'\t‣ Running py File for {gene_id}, {job_id}')

def evalModel(individual):
    gene_id = individual[0]
    # Initially, we don't have a fitness value
    return None

def check4model2run(gene_id):
    print(f'Checking for: SOTA_ROOT {VARIANT_DIR}/{MODEL}_{gene_id}.py')
    check_exist = f'{VARIANT_DIR}/{MODEL}_{gene_id}.py'
    
    # Check if the model file exists
    if not os.path.isfile(check_exist):
        print(f'\t☠ Model file does not exist for gene_id: {gene_id}, skipping evaluation.')
        GLOBAL_DATA[gene_id]['status'] = 'completed'
        GLOBAL_DATA[gene_id]['fitness'] = INVALID_FITNESS_MAX
        return

    model_path = os.path.join(OUTPUT_DIR, str(GENERATION), f'{gene_id}_model.txt')

    # Proceed only if the model hasn't been evaluated yet
    if GLOBAL_DATA[gene_id]['status'] != 'running eval':
        submit_run(gene_id)

    print(f'Checking for: SOTA_ROOT {VARIANT_DIR}/{MODEL}_{gene_id}.py')
    model_path = f'{VARIANT_DIR}/{MODEL}_{gene_id}.py'
    if os.path.exists(model_path):
        if GLOBAL_DATA[gene_id]['status'] != 'running eval':
            submit_run(gene_id)

def check4results(gene_id):
    def check4error(gene_id):
        job_id = GLOBAL_DATA[gene_id]['results_job']
        if GLOBAL_DATA[gene_id]['local_output'] is not None:
            state = check_contents_for_error(GLOBAL_DATA[gene_id]['local_output'])
            if state is None:
                print(GLOBAL_DATA[gene_id]['local_output'], flush=True) 
                raise Exception('Unexpected output from job')
            else:
                return state
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
                    return state
        return None
                
    job_done = check4error(gene_id)
    if job_done is True:
        out_dir = os.path.join(OUTPUT_DIR, str(GENERATION))
        # The job saves the model results to a file f'{gene_id}_results.csv'
        # results_path = os.path.join(out_dir, f'{gene_id}_results.csv')
        results_path = f'{SOTA_ROOT}/results/{gene_id}_results.csv'
        with open(results_path, 'r') as file:
            lines = file.readlines()
        # Skip header line (first line) and parse data line (second line)
        results = lines[-1].strip() if len(lines) > 1 else lines[0].strip()
        results = results.split(',')
        fitness = [float(r.strip()) for r in results]
        # TODO: get all features later
        fitness = [fitness[i] for i in range(len(FITNESS_WEIGHTS))]
        fitness = tuple(fitness)
        
        GLOBAL_DATA[gene_id]['status'] = 'completed'
        GLOBAL_DATA[gene_id]['fitness'] = fitness
        # print(f'Model from Gene: {gene_id} Evaluated')
    elif job_done is False:
        GLOBAL_DATA[gene_id]['status'] = 'completed'
        GLOBAL_DATA[gene_id]['fitness'] = INVALID_FITNESS_MAX
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
                ind.fitness.values = INVALID_FITNESS_MAX # Max error
                GLOBAL_DATA[gene_id]['status'] == "completed"      
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
                        ind.fitness.values = INVALID_FITNESS_MAX 
                        GLOBAL_DATA[gene_id]['status'] = 'completed'
                
                
                if GLOBAL_DATA[gene_id]['status'] == "completed":
                    # Process results and assign fitness
                    fitness_tuple = GLOBAL_DATA[gene_id]['fitness']  # Implement this function
                    ind.fitness.values = fitness_tuple
                elif time.time() - GLOBAL_DATA[gene_id].get('eval_start_time', GLOBAL_DATA[gene_id]['start_time']) > timeout:
                    print(f"Timeout for gene ID {gene_id} after {timeout} seconds without evaluation result")
                    cancel_eval_job(gene_id, f"no evaluation result for {timeout} seconds")
                    ind.fitness.values = INVALID_FITNESS_MAX 
                    GLOBAL_DATA[gene_id]['fitness'] = INVALID_FITNESS_MAX
                    GLOBAL_DATA[gene_id]['status'] = 'FAILED: TIMEOUT'
                else:
                    if 'results_job' not in GLOBAL_DATA[gene_id].keys():
                        ind.fitness.values = INVALID_FITNESS_MAX # Max error
                        print(f'\t☠ No Placeholder Fitness for: {gene_id}')
                        GLOBAL_DATA[gene_id]['status'] == "completed"
                    else:
                        print(f"\t‣ Still Waiting On: Gene: {gene_id}", flush=True)
                        print_job_info(GLOBAL_DATA[gene_id])
                        all_done = False  # Some jobs are still running
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
        if old_gene_id is not None and old_gene_id in GLOBAL_DATA.keys():
            del GLOBAL_DATA[old_gene_id]
        print(f'\t☑ {operation}: {new_gene_id}')
    else:
        print(f'\t☠ Failed {operation}: {new_gene_id}')
        if new_gene_id in GLOBAL_DATA.keys():
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
                print(f'Delayed Mating Check: {new_gene_id}, LLM Job ID: {job_id}')
                print(f'\t‣ Checking for Crossover Job Completion: {job_id} for {new_gene_id}')
                job_done = check4job_completion(job_id, extension="evolution/")

                if job_done:
                    print(f'\t‣ Model Files for {new_gene_id} are Loaded', flush=True) 
                else: 
                    print(f'\t‣ Error Loading Model Files for {new_gene_id}!!', flush=True)

                failed_process = not (successful_sub_flag and job_done)
                # Same as delayed_mutate_check: on failure update_individual drops
                # the child and restores the parent.
                individual = update_individual(individual, new_gene_id, old_gene_id=LINKED_GENES[k],
                                               process_success=not failed_process, process_type='Mating')

    return offspring

def delayed_creation_check(offspring):
    if DELAYED_CHECK is True:
        for individual in offspring:
            k = individual[0]
            if k in GLOBAL_DATA and GLOBAL_DATA[k]["status"] == "DELAYED_CHECK":
                GLOBAL_DATA[k]["status"]="subbed file"
                successful_sub_flag = GLOBAL_DATA[k]["sub_flag"]
                if successful_sub_flag:
                    gene_id = k
                    job_id = GLOBAL_DATA[k]["job_id"]
                    print(f'Checking for Job Completion: {job_id} for {gene_id}', flush=True)
                    job_done = check4job_completion(job_id, extension="evolution/")
                  
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
                if successful_sub_flag:
                    new_gene_id = k
                    job_id = GLOBAL_DATA[k]["job_id"]
                    print(f'Delayed Mutation Check: {new_gene_id}, LLM Job ID: {job_id}', flush=True)
                    print(f'\t‣ Checking for Creation Job Completion: {job_id} for {new_gene_id}')
                    job_done = check4job_completion(job_id, extension="evolution/")
                    if job_done:
                        print(f'\t‣ Model Files for {new_gene_id} are Loaded') 
                    else: 
                        print(f'\t☠ Error Loading Model Files for {new_gene_id}')

                    failed_process = not (successful_sub_flag and job_done)
                    old_gene_id = LINKED_GENES[k]
                    individual = update_individual(individual, new_gene_id, old_gene_id=old_gene_id,
                                                   process_success=not failed_process, process_type='Mutation')
                  
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
                                    'status':'subbed file', 'fitness':None, 'start_time':time.time()}
        
        if DELAYED_CHECK:
            GLOBAL_DATA[new_gene_id]['status'] = 'DELAYED_CHECK'
            return new_gene_id, None
        
        if successful_sub_flag:
            print(f'\t‣ Checking for Crossover Job Completion: {job_id} for {new_gene_id}')
            job_done = check4job_completion(job_id, local_output, extension="evolution/")
            if job_done:
                print(f'\t‣ Model Files for {new_gene_id} are Loaded')
            else: 
                print(f'\t‣ Error Loading Model Files for {new_gene_id}!!')

        failed_process = True if (successful_sub_flag is False) or (job_done is False) else False
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
                                'status':'subbed file', 'fitness':None, 'start_time':time.time()}
    
    if DELAYED_CHECK:
        LINKED_GENES[new_gene_id] = individual[0]
        GLOBAL_DATA[new_gene_id]['status'] = 'DELAYED_CHECK'
        individual[0] = new_gene_id
        individual = creator.Individual([new_gene_id])
        return individual
    
    if successful_sub_flag:
        print(f'\t‣ Checking for Mutation Job Completion: {job_id} for {new_gene_id}')
        job_done = check4job_completion(job_id, local_output, extension="evolution/")
        if job_done:
            print(f'\t‣ Model Files for {new_gene_id} are Loaded')
        else: 
            print(f'\t☠ Error Loading Model Files for {new_gene_id}')
    
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
def save_checkpoint(gen, folder_name="checkpoints", global_path=None, checkpoint_data=None):
    os.makedirs(folder_name, exist_ok=True)

    if global_path is not None:
        os.makedirs(global_path, exist_ok=True)
        global_file = os.path.join(global_path, f'global_gen_{gen}.pkl')
        if os.path.exists(global_file):
            with open(global_file, "rb") as file:
                try:
                    stored_global_data = pickle.load(file)
                except EOFError:
                    stored_global_data = {}  
        else:
            stored_global_data = {}


        print("ASSIGNING STORED GLOBAL DATA")

        if stored_global_data:
            print("Global data found, updating to file")
            stored_global_data["GLOBAL_DATA"].update(GLOBAL_DATA)
            stored_global_data["GLOBAL_DATA_HIST"].update(GLOBAL_DATA_HIST)
            stored_global_data["GLOBAL_DATA_ANCESTRY"].update(GLOBAL_DATA_ANCESTRY)
        else:
            print("No global data found, saving to a new file")
            stored_global_data = {
                "GLOBAL_DATA": GLOBAL_DATA,
                "GLOBAL_DATA_HIST": GLOBAL_DATA_HIST,
                "GLOBAL_DATA_ANCESTRY": GLOBAL_DATA_ANCESTRY
            }

        with open(global_file, 'wb') as file:
            pickle.dump(stored_global_data, file)
        print(f'Global data saved as {global_file}')

    if checkpoint_data is None:
        checkpoint_data = {
            "population": population,
            "hof": hof, 
        }
    filename = os.path.join(folder_name, f'checkpoint_gen_{gen}.pkl')
    with open(filename, 'wb') as file:
        pickle.dump(checkpoint_data, file)
    print(f"Population data saved as {filename}")

GEN_PATTERNS = {
    "checkpoint": re.compile(r"^checkpoint_gen_(\d+)\.pkl$"),
    "global": re.compile(r"^global_gen_(\d+)\.pkl$"),
}

def extract_generation(filename, kind):
    """Parse generation number from a checkpoint/global filename; return None if it doesn't match."""
    matcher = GEN_PATTERNS[kind].match(os.path.basename(filename))
    return int(matcher.group(1)) if matcher else None

def load_checkpoint(folder_name="checkpoints", checkpoint_file=None, global_path="checkpoints", global_file=None):
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
        print(f"Loaded population data from {filepath}")
        start_gen = extract_generation(checkpoint_file, "checkpoint") + 1

    if global_file is None:
        global_candidates = [f for f in os.listdir(global_path) if extract_generation(f, "global") is not None]
        global_files = sorted(global_candidates, key=lambda f: extract_generation(f, "global"), reverse=True)
        global_file = global_files[0] if global_files else None
    if global_file:
        filepath = os.path.join(global_path, global_file)
        with open(filepath, 'rb') as file:
            global_data = pickle.load(file)
        print(f"Loaded global data from {filepath}")

    return population_data, start_gen, global_data

def true_nsga2(pop, k):
    pop = tools.selNSGA2(pop, len(pop)) # 10 diff
    # selTournamentDCD needs a multiple of 4, so round up and trim back to k
    # instead of rounding down (3 survivors used to select 0 offspring).
    k4 = -(-k // 4) * 4
    pop = k4 * pop
    new_pop = tools.selTournamentDCD(pop, k4) # mults of 4
    return new_pop[:k]

def create_population(n, llm_model):
    individual_func = partial(toolbox.individual, llm_model=llm_model)
    return tools.initRepeat(list, individual_func, n)

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
toolbox.register("select", true_nsga2)

# TODO: start using percent diff of train acc vs val test acc as an over fitt metric 

# 40398682
GEN_COUNT = -1
TOP_N_GENES = None
LINKED_GENES = {}
GLOBAL_DATA = {}
GLOBAL_DATA_HIST = {}
GLOBAL_DATA_ANCESTRY = {}
GLOBAL_DATA_ANCESTRY[MODEL] = {'GENES':[MODEL], 'MUTATE_TYPE':["CREATED"]}

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

    PROMPT_GLOB = resolve_prompt_glob(args.prompt_group)

    print(DNA_TXT)
    print("ISLAND llm_model: ", llm_model)
    print(f"Prompt glob: {PROMPT_GLOB}")

    if AVAILABLE_LLM_MODELS and (not llm_model or llm_model not in AVAILABLE_LLM_MODELS):
        print("Error in Island Generation: No LLM specified. Exiting script")
        exit(1)

    population_data, start_gen, global_data = load_checkpoint(folder_name=args.checkpoints, global_path=args.global_path)
    if population_data:
        box_print("CHECKPOINT LOADED")
        GLOBAL_DATA = global_data["GLOBAL_DATA"]
        GLOBAL_DATA_HIST = global_data["GLOBAL_DATA_HIST"]
        GLOBAL_DATA_ANCESTRY = global_data["GLOBAL_DATA_ANCESTRY"]
        population = population_data["population"]
        hof = population_data["hof"]
    else:
        # Create an initial population
        start_gen = 1
        box_print("CREATING POPULATION FROM SEED CODE")
        population = toolbox.population(n=start_population_size, llm_model=llm_model)
        box_print("Batch Checking Created Genes", print_bbox_len=60, new_line_end=False)
        delayed_creation_check(population)
        hof = tools.ParetoFront()

    # Evaluate the entire population
    for ind in population:
        ind.fitness.values = PLACEHOLDER_FITNESS
        
    check_and_update_fitness(population)
    # Evolution
    for gen in range(start_gen, num_generations if migration_gen == 0 else ((start_gen + migration_gen - 1) // migration_gen) * migration_gen + 1):
        GEN_COUNT = gen
        TOP_N_GENES = tools.selSPEA2(population, NUM_EOT_ELITES)
        box_print(f"STARTING GENERATION: {gen}", new_line_end=False)
        print_population(population, GLOBAL_DATA)
        box_print(f"Invalid Removal", print_bbox_len=60, new_line_end=False)
        # Remove individuals with placeholder fitness
        population = [ind for ind in population if ind.fitness.values != INVALID_FITNESS_MAX]

        '''
        IF POPULATION IS LESS THAN NUM_ELITE INDIVIDUALS, GENERATE MORE FROM SCRATCH
         * todo: clean up this code/consolidate with previous code *
        '''

        box_print("CURRENT POPULATION SIZE:", len(population))
        while len(population) < num_elites:
            print("MINIMUM NUMBER OF IND NOT ACHIEVED, TRYING AGAIN")
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
            population = [ind for ind in population if ind.fitness.values != INVALID_FITNESS_MAX]
            box_print("CURRENT POPULATION SIZE:", len(population))

        print_population(population, GLOBAL_DATA)
        # Select the next generation's parents
        box_print(f"Selection", print_bbox_len=60, new_line_end=False)
        # These bypass the mutation and cross-over so we dont lose them

        elites = tools.selSPEA2(population, num_elites)
        
        # Select the next generation's parents
        if len(population) < population_size:
            print(f"Selecting {len(population)} offspring")
            offspring = toolbox.select(population, len(population))
        else:
            print(f"Selecting {population_size} offspring")
            offspring = toolbox.select(population, population_size)
        
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
        for mutant in offspring:
            if random.random() < mutation_probability:
                toolbox.mutate(individual=mutant, llm_model=llm_model)
                del mutant.fitness.values
                
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
        GLOBAL_DATA_HIST.update(GLOBAL_DATA.copy())
        # Replace the old population with the offspring
        population[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        print_scores(population, FITNESS_WEIGHTS)
        hof.update(population)
        save_checkpoint(gen, folder_name=args.checkpoints, global_path=args.global_path)
        LINKED_GENES = {}
        # mutate x prompts
        # mutate_prompts()

        best_ind = tools.selBest(population, 1)[0]
        print(f"Finished Generation {gen}")
        print(f"Best Individual: {best_ind}")
        print(f"Best Fitness: {best_ind.fitness.values}")
        
    print("-- End of Era --")
