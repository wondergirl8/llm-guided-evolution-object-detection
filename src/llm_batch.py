"""Batch compatible LLM requests and use instruction-model chat formatting."""

import asyncio
import time


def prepare_prompts(prompts, tokenizer):
    if getattr(tokenizer, 'chat_template', None):
        return [[{'role': 'system', 'content':
                  'You are a code editor. Follow the requested output format exactly. '
                  'Return only the requested code or configuration, without analysis or explanations.'},
                 {'role': 'user', 'content': prompt}] for prompt in prompts]
    return prompts


async def generate_batch(pipeline, tokenizer, requests):
    groups = {}
    for index, request in enumerate(requests):
        key = (request['max_new_tokens'], request['temperature'], request['top_p'])
        groups.setdefault(key, []).append((index, request))
    outputs = [None] * len(requests)
    for (tokens, temperature, top_p), group in groups.items():
        started = time.monotonic()
        prompts = prepare_prompts([request['prompt'] for _, request in group], tokenizer)
        results = await asyncio.to_thread(pipeline, prompts, max_new_tokens=tokens,
                                         temperature=temperature, top_p=top_p,
                                         return_full_text=False)
        if len(results) != len(group):
            raise ValueError('LLM pipeline returned the wrong number of completions')
        elapsed = round(time.monotonic() - started, 2)
        for (index, _), result in zip(group, results):
            text = result[0].get('generated_text')
            if not isinstance(text, str):
                raise ValueError('LLM pipeline completion must be text')
            outputs[index] = {'generated_text': text, 'response_time_sec': elapsed,
                              'batch_size': len(group), 'max_new_tokens': tokens,
                              'temperature': temperature, 'top_p': top_p}
    return outputs
