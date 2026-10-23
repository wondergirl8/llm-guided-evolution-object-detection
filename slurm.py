import os
import shlex
import yaml

from src.cfg import constants


def replace_script_configuration(file_path, new_config):
    if not os.path.isabs(file_path):
        file_path = os.path.join(constants.ROOT_DIR, file_path)
    with open(file_path, 'w') as f:
        f.write(new_config if new_config.endswith('\n') else new_config + '\n')


def save_to_yaml(llm, python, gpu, islands, file_path=constants.SLURM_CONFIG_DIR):
    yaml_data = {
        "gpu_selection": gpu,
        "python_bash_script": python,
        "llm_bash_script": llm,
        "islands_bash_script": islands,
    }
    with open(f'{file_path}/slurm_config.yaml', 'w') as f:
        yaml.dump(yaml_data, f, indent=4)


def parse_config_sections(content):
    """Parse the configuration file into named sections."""
    sections = {}
    current_name = None
    current_lines = []
    in_section = False
    
    for line in content:
        stripped = line.strip()
        if stripped == "------":
            if in_section:
                # End of section content
                sections[current_name] = "\n".join(current_lines)
                current_lines = []
                in_section = False
                current_name = None
            else:
                # Start of section content
                in_section = True
        elif not in_section and stripped:
            # This is a section name
            current_name = stripped
        elif in_section:
            current_lines.append(line)
    
    return sections


if __name__ == "__main__":
    configuration_path = os.path.join(constants.SLURM_CONFIG_DIR, f"{constants.CLUSTER}.txt")

    with open(configuration_path, 'r') as file:
        content = [item.strip() for item in file.readlines()]

    sections = parse_config_sections(content)

    # Generate run.sh
    run_sh = sections.get("run.sh", "") + f"""
echo "launching LLM Guided Evolution"
hostname
module load uv

export UV_CACHE_DIR="${{TMPDIR:-${{SLURM_TMPDIR:-/tmp}}}}/uv-cache-${{SLURM_JOB_ID:-$$}}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

export SERVER_HOSTNAME=$(hostname)
uv run python run_improved.py --checkpoints {shlex.quote(constants.OUTPUT_DIR)} --global_path {shlex.quote(constants.OUTPUT_DIR)}
"""
    replace_script_configuration("run.sh", run_sh)

    # Generate src/mixt.sh
    mixt_sh = sections.get("mixt.sh", "") + f"""
echo "Launching AIsurBL"
hostname
module load gcc/13.2.0
module load uv
source ~/.bashrc
export TOKENIZERS_PARALLELISM=false
export UV_CACHE_DIR="${{TMPDIR:-${{SLURM_TMPDIR:-/tmp}}}}/uv-cache-${{SLURM_JOB_ID:-$$}}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"
uv run python llm_crossover.py '{constants.SLURM_MIXT_INPUT_X}' '{constants.SLURM_MIXT_INPUT_Y}' '{constants.SLURM_MIXT_OUTPUT}'  --top_p {constants.SLURM_MIXT_TOP_P}   --temperature {constants.SLURM_MIXT_TEMPERATURE} --apply_quality_control '{constants.SLURM_MIXT_APPLY_QUALITY_CONTROL}' --bit {constants.SLURM_MIXT_BIT}
"""
    replace_script_configuration("src/mixt.sh", mixt_sh)

    # Get LLM GPU constraint
    llm_gpu = sections.get("llm-gpu", "").strip()

    # These recovery families must retain the device/MIG assignment from Slurm.
    # Keep the existing template behavior for other experiment families.
    cuda_visibility = (
        '# Preserve CUDA_VISIBLE_DEVICES assigned by Slurm.'
        if getattr(constants, 'SEED_FAMILY', None) in ('yolo26_p2', 'dfine')
        else 'export CUDA_VISIBLE_DEVICES=0'
    )

    # Generate python evaluation script template
    python_script = sections.get("python-bash-script", "") + f"""
echo "Launching Python Evaluation"
hostname

module load cuda
module load uv
{cuda_visibility}
unset VIRTUAL_ENV
export UV_CACHE_DIR="${{TMPDIR:-${{SLURM_TMPDIR:-/tmp}}}}/uv-cache-${{SLURM_JOB_ID:-$$}}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

# Run Python script
{{}}
"""

    # Generate LLM bash script template
    llm_script = sections.get("llm-bash-script", "") + f"""
echo "Launching AIsurBL"
hostname

module load cuda
module load uv
{cuda_visibility}
export UV_CACHE_DIR="${{TMPDIR:-${{SLURM_TMPDIR:-/tmp}}}}/uv-cache-${{SLURM_JOB_ID:-$$}}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

# Run Python script
{{}}
"""

    # Generate islands bash script template (for islands_wrapper.py)
    islands_script = sections.get("islands", sections.get("island-controller", "")) + """
cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"

module load cuda

export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/

# Run Python script
uv run python run_improved.py --checkpoints {} --global_path {} --llm_model {} --prompt_group {}
"""

    # Generate server.sh
    local_llm_server = f"""
echo "launching LLM Server"

# Optional chained submission count to work around walltime limits
COUNT=${{1:-1}}

hostname

module load cuda
module load uv

# Respect the GPU visibility selected by Slurm. Hard-coding 0,1 can make
# PyTorch probe devices outside the allocation on some cluster GPU nodes.
echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-<unset>}}"
export USE_TF=0
export USE_TORCH=1
export UV_CACHE_DIR="${{TMPDIR:-${{SLURM_TMPDIR:-/tmp}}}}/uv-cache-${{SLURM_JOB_ID:-$$}}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

export SERVER_HOSTNAME=$(hostname)

HOSTNAME_FILE=$(pwd)"/hostname.log"

# Write hostname to file so tests can find the server
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"
echo "Wrote hostname to $HOSTNAME_FILE"

# Log the island controller setting for debugging
echo "SUBMIT_ISLAND_CONTROLLER=${{SUBMIT_ISLAND_CONTROLLER:-<not set>}}"

# Default behavior: START island controller unless explicitly disabled
if [ "${{SUBMIT_ISLAND_CONTROLLER:-1}}" = "1" ]; then
    # Submit the paired island-controller job from here so the two stay in sync.
    echo "Submitting island controller (count=$COUNT)"
    sbatch island_controller.sbatch "$COUNT" "$SLURM_JOB_ID"
else
    echo "Skipping island controller submission (SUBMIT_ISLAND_CONTROLLER=${{SUBMIT_ISLAND_CONTROLLER}})"
fi

PORT=$(uv run python -c "from src.cfg.constants import PORT; print(PORT)")
echo "Serving on port $PORT"
uv run python -m uvicorn server:app --host $SERVER_HOSTNAME --port $PORT --workers 1
"""
    server_config = sections.get("server-sh", "")
    replace_script_configuration("server.sh", server_config + local_llm_server)
    print(f"Generated server.sh with config:\n{server_config}")

    # Generate unified island_controller.sbatch
    island_controller = sections.get("island-controller", sections.get("islands", "")) + f"""
# Island controller arguments:
# $1 (COUNT): Number of remaining restart iterations
# $2 (PREV_SERVER_JOB_ID): Job ID of the currently-running server (to cancel)
COUNT=$1
PREV_SERVER_JOB_ID=${{2:-}}

cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"
echo "$COUNT iterations remaining"

module load cuda

# LLM_Storage
export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/

# Change to the repository root
cd {constants.ROOT_DIR}

uv run python islands_wrapper.py {constants.ISLAND_CONTROLLER_RUN_NAME} \\
    --num_islands {constants.ISLAND_CONTROLLER_NUM_ISLANDS} \\
    --llms {constants.ISLAND_CONTROLLER_LLMS} \\
    --prompt_groups "{constants.ISLAND_CONTROLLER_PROMPT_GROUPS}"

if (( COUNT > 1 )); then
    NEXT_COUNT=$((COUNT - 1))
    if [[ -n "$PREV_SERVER_JOB_ID" ]]; then
        echo "Stopping previous server job: $PREV_SERVER_JOB_ID"
        scancel "$PREV_SERVER_JOB_ID"
    fi
    echo "Controller completed; launching next server iteration with count: $NEXT_COUNT"
    sbatch server.sh "$NEXT_COUNT"
fi
"""
    island_controller = island_controller.replace(
        "#SBATCH --job-name=Islands",
        "#SBATCH --job-name=IslandsController"
    )
    replace_script_configuration("island_controller.sbatch", island_controller)
    print(f"Generated island_controller.sbatch")

    # Save templates to YAML for runtime use
    save_to_yaml(llm_script, python_script, llm_gpu, islands_script)
    print(f"Saved configuration to {constants.SLURM_CONFIG_DIR}/slurm_config.yaml")
