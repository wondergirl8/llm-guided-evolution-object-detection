"""Small runtime checks for an allocated local LLM server."""
import os
import re
import subprocess
import time


SLURM_STATUS_ATTEMPTS = 3


def slurm_status(command, attempts=SLURM_STATUS_ATTEMPTS):
    """Retry temporary scheduler transport errors, never a reported job state."""
    for attempt in range(attempts):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            if attempt == attempts - 1:
                raise RuntimeError(f'Slurm status request timed out after {attempts} attempts')
        else:
            if result.returncode == 0:
                return result
            message = (result.stderr + result.stdout).lower()
            transient = ('timed out' in message or
                         'backup controller in standby mode' in message)
            if not transient:
                return result
            if attempt == attempts - 1:
                return result
        time.sleep(2)


def gpu_memory_budget(torch):
    """Actually initialize every visible device before handing GPUs to HF.

    An NVML-derived device count is not proof that CUDA can use those devices.
    Never alter CUDA_VISIBLE_DEVICES or silently drop an inaccessible GPU.
    """
    if not torch.cuda.is_available():
        raise RuntimeError('LLM server requires usable CUDA GPUs')
    count = torch.cuda.device_count()
    expected = int(os.getenv('LLMGE_EXPECTED_GPUS', str(count)))
    if count < 1 or count != expected:
        raise RuntimeError(f'Expected {expected} allocated GPUs, but PyTorch reports {count}')
    budget = {}
    for index in range(count):
        try:
            properties = torch.cuda.get_device_properties(index)
            probe = torch.empty(1, device=f'cuda:{index}')
            torch.cuda.synchronize(index)
            free, total = torch.cuda.mem_get_info(index)
            del probe
            if free <= 0 or total <= 0:
                raise RuntimeError('GPU reports no usable memory')
        except Exception as error:
            raise RuntimeError(f'Allocated GPU {index}/{count} cannot initialize; '
                               'refusing LLM startup on this allocation') from error
        print(f'GPU {index}: {properties.name}; free={free}; total={total}; CUDA probe passed', flush=True)
        budget[index] = int(free * .90)
    return budget


def assert_llm_job_active():
    """Fail the controller when its explicitly identified server job dies."""
    job = os.getenv('LLMGE_SERVER_JOB_ID')
    if not job:
        return
    if not re.fullmatch(r'\d+', job):
        raise RuntimeError('LLMGE_SERVER_JOB_ID must be a numeric Slurm job ID')
    result = slurm_status(['scontrol', 'show', 'job', job, '-o'])
    if result.returncode:
        raise RuntimeError(f'Cannot verify LLM server job {job}; stopping: {result.stderr.strip()}')
    match = re.search(r'(?:^|\s)JobState=(\S+)', result.stdout)
    if not match or match[1] not in ('PENDING', 'CONFIGURING', 'RUNNING', 'SUSPENDED'):
        state = match[1] if match else 'UNKNOWN'
        raise RuntimeError(f'LLM server job {job} is {state}; stopping instead of waiting repeatedly')
