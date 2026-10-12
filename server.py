import asyncio
import os
import threading
import time

# This server is PyTorch-only. Prevent Transformers from importing TensorFlow
# pipeline modules, which can fail when TensorFlow is installed but incomplete.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import torch
import transformers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.cfg.constants import *
from src.llm_batch import generate_batch

app = FastAPI(title="LLM API", version="1.0")

BATCH_SIZE = 8  # num of LLM requests to process at once
BATCH_WAIT_TIME = 2  # max wait time for batch to fill in s

class LLMRequest(BaseModel):
    prompt: str
    max_new_tokens: int = LLM_MAX_NEW_TOKENS
    top_p: float = 0.8
    temperature: float = 0.7

class LLMModel:
    _instance = None
    _lock = threading.Lock()

    @staticmethod
    def _log_cuda_state():
        print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
        print(f"torch version={torch.__version__}", flush=True)
        try:
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count()
        except Exception as err:
            print(f"torch cuda introspection failed={err}", flush=True)
            return

        print(f"torch cuda available={cuda_available}", flush=True)
        print(f"torch cuda device count={device_count}", flush=True)

    @staticmethod
    def _validate_cuda_available():
        LLMModel._log_cuda_state()
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError(
                "CUDA is not available to PyTorch, so the LLM server would run on CPU. "
                "Check the Slurm GPU request, CUDA module, and uv/torch environment."
            )
        count = torch.cuda.device_count()
        expected = int(os.getenv('LLM_SERVER_EXPECTED_GPUS', '0'))
        if expected and count != expected:
            raise RuntimeError(f'Expected {expected} allocated CUDA devices, but PyTorch sees {count}. '
                               'Check Slurm GPU visibility and any MIG restrictions.')
        # NVML visibility alone does not prove CUDA can initialize a device.
        for index in range(count):
            properties = torch.cuda.get_device_properties(index)
            print(f'CUDA device {index}: {properties.name}; '
                  f'{properties.total_memory / 2**30:.1f} GiB', flush=True)
            probe = torch.empty(1, device=f'cuda:{index}')
            torch.cuda.synchronize(index)
            del probe
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print(f"Loading model at {MODEL_PATH} for the first time", flush=True)
                    cls._instance = super(LLMModel, cls).__new__(cls)
                    cls._instance._initialize()
                    print('I created my instance', flush=True)
                    print(dir(cls._instance), flush=True)            
        return cls._instance
    
    def _initialize(self):
        print("initializing", flush=True)
        self._validate_cuda_available()
        # TODO figure out how to better handle the initialization (i.e. mixtral dies because it doesn't have attention)
        # TODO find out why when this dies the code around it continues i.e. a model is returned to generate_text, but I never see the print out of "I created my instance"
        # Resolve HF cache directories: if MODEL_PATH is a HF cache dir
        # (contains 'snapshots/'), find the actual model snapshot
        model_path = MODEL_PATH
        import os
        snapshots_dir = os.path.join(model_path, 'snapshots')
        if os.path.isdir(snapshots_dir):
            # Use the latest snapshot
            snapshot_dirs = sorted(os.listdir(snapshots_dir))
            if snapshot_dirs:
                model_path = os.path.join(snapshots_dir, snapshot_dirs[-1])
                print(f"Resolved HF cache to snapshot: {model_path}")

        device_count = torch.cuda.device_count()
        max_memory = {idx: "120GiB" for idx in range(device_count)} if device_count else None
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            attn_implementation="sdpa" # faster inference
        ).eval()
        print("model loaded", flush=True)
        hf_device_map = getattr(self.model, "hf_device_map", None)
        if hf_device_map is not None:
            print(f"model device map={hf_device_map}", flush=True)
            if all(str(device) == "cpu" for device in hf_device_map.values()):
                raise RuntimeError(
                    "Transformers placed the entire model on CPU despite CUDA being available."
                )
        else:
            first_param_device = next(self.model.parameters()).device
            print(f"model first parameter device={first_param_device}", flush=True)
            if first_param_device.type == "cpu":
                raise RuntimeError(
                    "Transformers placed the model on CPU despite CUDA being available."
                )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
        print("tokenizer created", flush=True)
        
        # decoder-only models need left padding for correct generation order
        self.tokenizer.padding_side = "left"

        # for batching, need to set pad tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            task="text-generation",
            temperature=0.1,
            top_p=0.15,
            top_k=0,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            repetition_penalty=1.1,
            do_sample=True,
            batch_size=BATCH_SIZE # for batch support
        )
        print("pipeline created", flush=True)
        
        self.request_queue = asyncio.Queue() # queue for holding requests to process
        self.batch_task = None # current task
        self.batch_lock = asyncio.Lock() # lock for
        self.batch_id = 0
        self.is_processing = False # current state
        print("ready to go", flush=True)
    
    async def start_batch_processor(self):
        """Start the batch processor if it's not already running"""
        async with self.batch_lock:
            if not self.is_processing:
                self.is_processing = True
                self.batch_task = asyncio.create_task(self._batch_processor())
    
    async def _batch_processor(self):
        """Process requests in batches"""
        try:
            while True:
                batch = []
                futures = []
                
                try:
                    # check queue for requests
                    request, future = await self.request_queue.get()
                    batch.append(request)
                    futures.append(future)
                    
                    # try to fill the batch until BATCH_SIZE or timeout
                    batch_start_time = time.time()
                    while len(batch) < BATCH_SIZE and (time.time() - batch_start_time) < BATCH_WAIT_TIME:
                        try:
                            req, fut = await asyncio.wait_for(
                                self.request_queue.get(),
                                timeout=max(0, BATCH_WAIT_TIME - (time.time() - batch_start_time))
                            )
                            batch.append(req)
                            futures.append(fut)
                        except asyncio.TimeoutError:
                            break
                    
                    batch_size = len(batch)
                    self.batch_id += 1
                    batch_id = self.batch_id
                    print(
                        f"Processing batch {batch_id} of {batch_size} requests "
                        f"(queued after collect: {self.request_queue.qsize()})",
                        flush=True,
                    )
                    
                    outputs = await generate_batch(self.pipeline, self.tokenizer, batch)
                    print(f"Finished batch {batch_id} of {batch_size} requests", flush=True)
                    for output, future in zip(outputs, futures):
                        if not future.done():
                            future.set_result(output)
                except Exception as e:
                    print(f"Error processing batch: {str(e)}", flush=True)
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)
                finally:
                    # Exactly once even if delivery fails after some futures resolve.
                    for _ in futures:
                        self.request_queue.task_done()

                # Check if queue is empty, if so - sleep briefly to save resources
                if self.request_queue.empty():
                    await asyncio.sleep(0.01)
        
        except asyncio.CancelledError:
            print("Batch processor cancelled", flush=True)
        except Exception as e:
            print(f"Unexpected error in batch processor: {str(e)}", flush=True)
        finally:
            async with self.batch_lock:
                self.is_processing = False
    
    async def generate(self, request_dict):
        """Submit a request to the batch processor"""
        # future is a placeholder for later result
        future = asyncio.Future()
        
        await self.request_queue.put((request_dict, future))
        print(f"Request queued (queue size: {self.request_queue.qsize()})", flush=True)
        
        # start processing batches if not already started
        await self.start_batch_processor()
        
        # wait & return future result
        return await future

@app.post("/generate")
async def generate_text(request: LLMRequest):
    """
    Submits LLMRequest to the local model. If the model has not been intialized before, this function will first have to intialize the model.

    Parameters:
    LLMRequest:
        txt2llm (str): input to llm
        max_new_tokens (int): maximum number of tokens model should generate
        top_p (int): threshold, higher to consider wider range of words
        temperature (int): randomness, higher for more varied outputs

    Returns:
    dict: generated_text (output of LLM) and response_time (time after intializing model until generated text obtained)
    """
    try:
        # Convert request to dict
        request_dict = {
            "prompt": request.prompt,
            "max_new_tokens": request.max_new_tokens,
            "top_p": request.top_p,
            "temperature": request.temperature
        }
        
        # Get the model instance
        model = LLMModel()
        # Submit to the batch processor and wait for result
        start_time = time.time()
        print(f"Request received at {time.strftime('%H:%M:%S', time.localtime(start_time))}", flush=True)
        
        result = await model.generate(request_dict)
        
        print(f"Request completed in {time.time() - start_time:.2f}s", flush=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "LLM API is running!", "slurm_job_id": os.getenv("SLURM_JOB_ID")}

print('Server running with server-side batching!', flush=True)


@app.on_event("startup")
async def preload_model():
    """Load the model at process start so first request is fast."""
    model = LLMModel()
    await model.start_batch_processor()
