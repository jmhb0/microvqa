"""
python -m ipdb models/openai_api.py 
Functions for calling api
Needs to have set OPENAI_API_KEY.
Models: https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
"""
import ipdb
import base64
import asyncio
import openai
from pathlib import Path
from PIL import Image
import numpy as np
import os
import io
import shutil
from typing import List, Tuple
import lmdb
import json
import sys
import logging
import concurrent.futures
from threading import Lock
from functools import lru_cache
import time
from tqdm import tqdm
import pickle

import sys

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

sys.path.insert(0, "..")
sys.path.insert(0, ".")
from cache import cache_utils


cache_demo = lmdb.open("cache/cache_demo", map_size=int(1e12))
cache_demo_full = lmdb.open("cache/cache_demo_full",
                              map_size=int(1e12))  # workaround
cache_lock = Lock()

HITS = 0
MISSES = 0

def call_gpt(
    # args for setting the `messages` param
    text: str,
    imgs: List[np.ndarray] = None,
    system_prompt: str = None,
    conversation: List = None,  # fails if there's images right now
    json_mode: bool = False,
    response_format: str = None,
    # kwargs for client.chat.completions.create
    detail: str = "high",
    model: str = "gpt-4o-2024-08-06",
    # model: str ="gpt-4o-mini-2024-07-18",
    temperature: float = 1,
    max_tokens: int = 2048,
    top_p: float = 1,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    seed: int = 0,
    n: int = 1,
    # args for caching behaviour
    cache: bool = True,
    overwrite_cache: bool = False,
    debug=None,
    verbose=True,
    api='openai',
    # if json_mode=True, and not json decodable, retry this many time
    num_retries: int = 3):
    """ 
    Call GPT LLM or VLM synchronously with caching.
    To call this in a batch efficiently, see func `call_gpt_batch`.

    If `cache=True`, then look in database ./cache/cache_demo for these exact
    calling args/kwargs. The caching only saves the first return message, and not
    the whole response object. 

    imgs: optionally add images. Must be a sequence of numpy arrays. 
    overwrite_cache (bool): do NOT get response from cache but DO save it to cache.
    seed (int): doesnt actually work with openai API atm, but it is in the 
        cache key, so changing it will force the API to be called again
    """

    # config the client based on `api` parameter
    if api == "openai":
        base_url = "https://api.openai.com/v1"
        api_key = os.getenv("OPENAI_API_KEY")
    elif api in ("openrouter", "openrouter"):
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.getenv("OPENROUTER_API_KEY")
    elif api == "hyperbolic":
        base_url = "https://api.hyperbolic.xyz/v1"
        api_key = os.getenv("HYPERBOLIC_API_KEY")
    else:
        raise NotImplementedError()
    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    global HITS, MISSES

    if verbose:
        print(f"\rGPT cache. Hits: {HITS}. Misses: {MISSES}", end="")

    # response format
    if response_format:
        assert not json_mode
        response_format_kwargs = response_format.schema()
    else:
        if json_mode:
            response_format_kwargs = {"type": "json_object"}
        else:
            response_format_kwargs = {"type": "text"}

    if conversation:
        messages = conversation
    else:
        messages = [{
            "role": "system",
            "content": system_prompt,
        }] if system_prompt is not None else []

    # text prompt
    content = [
        {
            "type": "text",
            "text": text,
        },
    ]

    # for imgs, put a hash key representation in content for now. If not cahcing,
    # we'll replace this value later (it's because `_encode_image_np` is slow)
    if imgs:
        content.append(
            {"imgs_hash_key": [cache_utils.hash_array(im) for im in imgs]})

    # text & imgs to message - assume one message only
    messages.append({"role": "user", "content": content})

    # arguments to the call for client.chat.completions.create
    kwargs = dict(
        messages=messages,
        response_format=response_format_kwargs,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        n=n,
    )
    if "o1" in model:
        del kwargs['max_tokens']

    if cache:
        cache_key = json.dumps(kwargs, sort_keys=True)
        with cache_lock:
            msg = cache_utils.get_from_cache(cache_key, cache_demo)
        if msg is not None and not overwrite_cache:
            if json_mode or response_format:
                msg = json.loads(msg)
            with cache_lock:
                HITS += 1
            conversation = messages + [{"role": "assistant", "content": msg}]

            # if the response object was also saved incache_demo_full, get it
            with cache_lock:
                response_cache = cache_utils.get_from_cache(
                    cache_key, cache_demo_full)
            if response_cache is not None:
                response_cache = json.loads(response_cache)

            return msg, response_cache, messages, conversation
    with cache_lock:
        MISSES += 1
        # print("Debug: ", debug)

    # not caching, so if imgs,then encode the image to the http payload
    if imgs:
        assert "imgs_hash_key" in content[-1].keys()
        content.pop()

        if api != 'openai':
            imagelst = [Image.fromarray(im) for im in imgs]
            base64_imgs = ImageList(tuple(imagelst)).to_base64()
        else:
            base64_imgs = [_encode_image_np(im) for im in imgs]

        if 'Qwen' in model:
            base64_imgs = base64_imgs[:4]

        # if 'gemini' in model:
        #     size = get_base64_sizes(base64_imgs)
        #     pass

        for base64_img in base64_imgs:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}",
                    # "detail": detail,
                },
            })
    # not caching, so make the response format pydantic class again, not stringified version of it
    if response_format:
        kwargs['response_format'] = response_format

    # if 'anthropic' not in model:
    response = client.beta.chat.completions.parse(**kwargs)
    # else:
    # response = client.chat.completions.create(**kwargs)

    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    msg = response.choices[0].message.content

    response_cache = dict(msg=msg,
                          prompt_tokens=prompt_tokens,
                          completion_tokens=completion_tokens)
    response_cache['cost'] = compute_api_call_cost(prompt_tokens,
                                                   completion_tokens,
                                                   model=model)
    conversation = messages + [{"role": "assistant", "content": msg}]

    # save to cache if enabled
    if cache:
        with cache_lock:
            cache_utils.save_to_cache(cache_key, msg, cache_demo)
            cache_utils.save_to_cache(cache_key, json.dumps(response_cache),
                                      cache_demo_full)

    if json_mode or response_format:
        msg = json.loads(msg)

    return msg, response_cache, messages, conversation


def _encode_image_np(image_np: np.array):
    """ Encode numpy array image to bytes64 so it can be sent over http """
    assert image_np.ndim == 3 and image_np.shape[-1] == 3
    image = Image.fromarray(image_np)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def get_base64_sizes(base64_imgs):
    for i, img in enumerate(base64_imgs):
        # Get size of the base64 string in bytes
        size_bytes = len(img.encode('utf-8'))
        size_mb = size_bytes / (1024 * 1024)
        print(f"Image {i}: {size_mb:.2f} MB ({size_bytes:,} bytes)")

    total_bytes = sum(len(img.encode('utf-8')) for img in base64_imgs)
    return total_bytes


def call_gpt_batch(texts,
                   imgs=None,
                   seeds=None,
                   json_modes=None,
                   get_meta=True,
                   debug=None,
                   num_threads=64,
                   **kwargs):
    """ 
    with multithreading
    if return_meta, then return a dict that tells you the runtime, the cost

    kwargs gets forwarded to `call_gpt`
    """
    n = len(texts)
    if imgs is None:
        imgs = [None] * n

    assert n == len(imgs), "texts and imgs must have the same length"

    # handle having a different seed per call
    all_kwargs = [kwargs.copy() for _ in range(n)]
    if seeds is not None or json_modes is not None or debug is not None:
        for i in range(n):
            if seeds is not None:
                all_kwargs[i]['seed'] = seeds[i]
            if json_modes is not None:
                all_kwargs[i]['json_mode'] = json_modes[i]
            if debug is not None:
                all_kwargs[i]['debug'] = debug[i]

    if num_threads == 1:
        # Regular sequential processing
        results = []
        for text, img, _kwargs in zip(texts, imgs, all_kwargs):
            result = call_gpt(text, img, **_kwargs)
            results.append(list(result))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for text, img, _kwargs in zip(texts, imgs, all_kwargs):
                future = executor.submit(call_gpt, text, img, **_kwargs)
                futures.append(future)
            results = [list(future.result()) for future in futures]

    # reset the cache logging
    global HITS, MISSES
    HITS, MISSES = 0, 0
    print()

    if get_meta:
        # compute cost in a way that handles the back-compatibility issue with old cache results
        for i in range(len(results)):
            msg, response = results[i][:2]
            if type(results[i][1]) is dict:
                results[i][1] = response['cost']
            else:
                results[i][1] = 0

    return results


def compute_api_call_cost(prompt_tokens: int,
                          completion_tokens: int,
                          model="gpt-4-turbo-2024-04-09"):
    """
    Warning: prices need to be manually updated from
    https://openai.com/api/pricing/
    """
    prices_per_million_input = {
        "o1": 15,
        "o1-mini": 3,
        "gpt-4o-mini": 0.15,
        "gpt-4o": 5,
        "gpt-4-turbo": 10,
        "gpt-4": 30,
        "gpt-3.5-turbo": 0.5,
        'anthropic/claude-3.5-sonnet': 3,
        'Qwen2-VL-72B-Instruct': 0.4,
        'Qwen2-VL-7B-Instruct' : 0.1,
        "google/gemini-pro-1.5": 1.25,
        "llama-3.2-90b-vision-instruct": 0.9,
        "llama-3.2-11b-vision-instruct":0.055,
        'anthropic/claude-3-opus' : 15,
        "pixtral-12b" : 0.15,
    }
    prices_per_million_output = {
        "o1": 60,
        "o1-mini": 12,
        "gpt-4o-mini": 0.075,
        "gpt-4o": 15,
        "gpt-4-turbo": 30,
        "gpt-4": 60,
        "gpt-3.5-turbo": 1.5,
        'anthropic/claude-3.5-sonnet': 15,
        'Qwen2-VL-72B-Instruct': 0.4,
        'Qwen2-VL-7B-Instruct' : 0.1,
        "google/gemini-pro-1.5": 5,
        "llama-3.2-90b-vision-instruct": 0.9,
        "llama-3.2-11b-vision-instruct":0.055,
        'anthropic/claude-3-opus' : 75,
        "pixtral-12b" : 0.15,
    }
    if "o1-preview" in model:
        key = "o1"
    elif "o1-mini" in model:
        key = "o1-mini"
    elif "gpt-4o-mini" in model:
        key = "gpt-4o-mini"
    elif "gpt-4o" in model:
        key = "gpt-4o"
    elif "gpt-4-turbo" in model:
        key = "gpt-4-turbo"
    elif 'gpt-4' in model:
        key = "gpt-4"
    elif 'gpt-3.5-turbo' in model:
        key = "gpt-3.5-turbo"
    elif 'claude-3.5-sonnet' in model:
        key = 'anthropic/claude-3.5-sonnet'
    elif 'claude-3-opus' in model:
        key = 'anthropic/claude-3-opus'
    elif 'Qwen2-VL-72B-Instruct' in model:
        key = 'Qwen2-VL-72B-Instruct'
    elif 'Qwen2-VL-7B-Instruct' in model:
        key = 'Qwen2-VL-7B-Instruct'
    elif 'gemini-pro-1.5' in model:
        key = 'google/gemini-pro-1.5'
    elif "llama-3.2-90b-vision-instruct" in model:
        key = "llama-3.2-90b-vision-instruct"
    elif "llama-3.2-11b-vision-instruct" in model:
        key = "llama-3.2-11b-vision-instruct"
    elif "pixtral-12b" in model:
        key = "pixtral-12b"
    
    else:
        return 0
        raise NotImplementedError(f"Did not record prices for model {model}")

    price = prompt_tokens * prices_per_million_input[
        key] + completion_tokens * prices_per_million_output[key]
    price = price / 1e6

    return price


class ImageList:
    """Handles a list of images with encoding support for base64 conversion.

    Attributes:
        images (Tuple[Image.Image]): A tuple containing PIL Image objects.
    """

    images: Tuple[Image.Image]

    def __init__(self, images):
        self.images = images

    @staticmethod
    @lru_cache()  # pickle strings are hashable and can be cached.
    def _encode(image_pkl: str) -> str:
        """Encodes a pickled image to a base64 string.

        Args:
            image_pkl (str): A serialized representation of the image.

        Returns:
            str: The base64-encoded PNG image.
        """
        image: Image.Image = pickle.loads(image_pkl)  # deserialize image

        with io.BytesIO() as buffer:
            image.save(buffer, format="jpeg")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def to_base64(self) -> Tuple[str]:
        """Converts the images in the list to base64-encoded PNG format.

        Returns:
            Tuple[str]: A tuple of base64-encoded strings for each image.
        """
        image_pkls = [pickle.dumps(img) for img in self.images]
        return tuple(ImageList._encode(pkl) for pkl in image_pkls)


def test_basic():
    model = "gpt-4o-mini"
    text = "How did Steve Irwin die? "
    overwrite_cache = True
    # overwrite_cache = False
    cache = True
    res = call_gpt(text,
                   model=model,
                   cache=cache,
                   overwrite_cache=overwrite_cache,
                   json_mode=False)
    msg = res[0]


def test_system_prompt():
    model = "gpt-4o-mini"
    system_prompt = "After answering any question, provide a short anecdote from the perspective of a bogan Australian visiting California."
    text = "How did Steve Irwin die? "
    overwrite_cache = True
    # overwrite_cache = False
    cache = True
    res = call_gpt(text,
                   system_prompt=system_prompt,
                   model=model,
                   cache=cache,
                   overwrite_cache=overwrite_cache,
                   json_mode=False)
    msg = res[0]


def test_conversation():
    model = "gpt-4o-mini"
    text0 = "How did Steve Irwin die? "
    cache = True
    res0 = call_gpt(text0, model=model, cache=cache, json_mode=False)
    msg0 = res0[0]

    print()
    print(msg0)
    print()
    conversation = res0[3]
    text_refined = "What are the last 5 words in your previous response?"

    res1 = call_gpt(text_refined,
                    conversation=conversation,
                    model=model,
                    cache=cache,
                    json_mode=False)
    msg1 = res1[0]
    print()
    print(msg1)
    print()
    pass


def test_response_format():
    from pydantic import BaseModel

    class Theories(BaseModel):
        theories: list[str]
        probabilities: list[float]

    model = "gpt-4o-mini"
    # model = "o1-mini" # o1-mini fails
    text0 = "Give 3 conspiracy theories about how Steve Irwin died. Describe briefly."
    cache = False

    res0 = call_gpt(text0,
                    model=model,
                    cache=cache,
                    json_mode=False,
                    response_format=Theories)
    msg0 = res0[0]
    print()
    print(msg0)


def test_batch():
    model = "gpt-4o-mini"
    text = "How did Steve Irwin die? "
    batch_texts = [text] * 10
    seeds = list(range(10))  # since the text is identical
    overwrite_cache = True
    overwrite_cache = False
    cache = True
    res = call_gpt_batch(batch_texts,
                         model=model,
                         cache=cache,
                         seeds=seeds,
                         overwrite_cache=overwrite_cache,
                         json_mode=False)
    msg = res[0]
    cost = sum([r[1] for r in res])
    print(f"Total cost ${cost:.3f}")


def test_claude():
    model = "anthropic/claude-3.5-sonnet"
    api = "openrouter"
    text = "What model is this? Who build you? Also, how did Steve Irwin die?"
    cache = False
    res0 = call_gpt(text, model=model, cache=cache, api=api, json_mode=False)
    msg = res0[0]
    ipdb.set_trace()
    pass


def test_gemini():
    model = "google/gemini-pro-1.5"
    api = "openrouter"
    text = "What model is this? Who build you? Also, how did Steve Irwin die?"
    cache = False
    res0 = call_gpt(text, model=model, cache=cache, api=api, json_mode=False)
    msg = res0[0]
    ipdb.set_trace()
    pass


def test_qwen():
    model = "Qwen/Qwen2-VL-72B-Instruct"
    api = "hyperbolic"
    text = "What model is this? Who build you? Also, how did Steve Irwin die?"
    # text = "Who makes the best clam chowder?"
    cache = False
    res0 = call_gpt(text, model=model, cache=cache, api=api, json_mode=False)
    msg = res0[0]
    ipdb.set_trace()
    pass


def test_llama():
    model = "meta-llama/llama-3.2-90b-vision-instruct"
    api = "openrouter"
    text = "What model is this? Who build you? Also, how did Steve Irwin die?"
    # text = "Who makes the best clam chowder?"
    cache = False
    res0 = call_gpt(text, model=model, cache=cache, api=api, json_mode=False)
    msg = res0[0]
    ipdb.set_trace()
    pass


def test_gemini_img():
    model = "google/gemini-pro-1.5"
    # random_image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    # imgs = [random_image, random_image, random_image]
    api = "openrouter"
    text = "What model is this? Who build you? Also, how did Steve Irwin die?"
    cache = False
    res0 = call_gpt(text,
                    imgs,
                    model=model,
                    cache=cache,
                    api=api,
                    json_mode=False)
    msg = res0[0]
    ipdb.set_trace()
    pass


# basic testing
if __name__ == "__main__":
    import time
    import sys
    sys.path.insert(0, "..")
    sys.path.insert(0, ".")

    # test_basic()
    # test_system_prompt()
    # test_batch()
    # test_conversation()
    # test_response_format()
    # test_claude()
    # test_gemini()
    # test_qwen()
    # test_llama()
    # test_gemini_img()
