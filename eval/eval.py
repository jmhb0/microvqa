"""
python -m ipdb eval/eval.py
"""

import ipdb
import datasets
import numpy as np
import re
from eval.openai_api import call_gpt_batch
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm
import datasets

# based on prompt from MMLU-pro https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/b7b9ffd84b2c21a5bfcf174fc65e5e6d74ca09a8/evaluate_from_api.py",
PROMPT_TEMPLATE = """\
The following is a multiple choice question (with answers). 
Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.

{{QUESTION}}

Options:
{{CHOICES}}
"""

PROMPT_TEMPLATE_NO_IMAGE = """\
The following is a multiple choice question (with answers).\
If an image is mentioned ignore this information and try your best to answer the question.
Think step by step and then output the answer in the format of \"The answer is (X)\" at the end."

{{QUESTION}}

Options:
{{CHOICES}}
"""

## this regex works for "the answer is (3)" or "the answer is 3"
# REGEX_PATTERN = r"answer is \(?([0-9])\)?"
## this regex works for "the answer is (3)" or "the answer is 3" or "the answer is **(3)**". The latter case happened sometimes for gpt-4o and  grok and maybe some others too
REGEX_PATTERN = r"answer is \*?\*?\(?([0-9])\)?\*?\*?"

def eval_qa(dataset: datasets.Dataset,
            model: str ='gpt-4o-2024-11-20',
            no_image: bool = False,
            seed: int = 0,
            results_dir: str = None,
            num_threads: int = 64,
            is_rate_limited: bool = False, 
            is_stage1 = False,
            prompts_preloads: list = None):
    """ 
    Args:
        dataset (datasets.Dataset): The dataset with questions and choices for evaluation.
        model (str): The model to use, default is 'gpt-4o-2024-11-20'.
        no_image (bool): If True, evaluation runs without images.
        seed (int): Random seed for reproducibility.
        results_dir (str): Directory to save results.
        num_threads (int): Number of threads for parallel processing.
        is_rate_limited (bool): Indicates if the model is subject to rate limits.
        prompts_preloads (list): Optional preloaded prompts and images, since it's timeconsuming. 
            If not provided, will generate. Once function is done, it will return the prompts for reuse.
	"""

    batch_prompts_text = []
    batch_prompts_imgs = []
    gts = []

    print(f"Preparing all the prompts")
    if prompts_preloads is None:

        for i, row in tqdm(enumerate(dataset), total=len(dataset)):
            if no_image:
                prompt = PROMPT_TEMPLATE_NO_IMAGE
            else:
                prompt = PROMPT_TEMPLATE
            if not is_stage1:
                question = row['question']
                choices = row['choices']
                correct_index = row['correct_index']
            else:
                question = row['question_1']
                choices = row['choices_1']
                correct_index = row['correct_index_1']
                
            prompt = prompt.replace("{{QUESTION}}", question)
            choices_str = ""
            for j, ch in enumerate(choices):
                choices_str += f"  ({j+1}): {ch}\n" # base-1 indexing
            prompt = prompt.replace("{{CHOICES}}", choices_str)
            batch_prompts_text.append(prompt)
            gts.append(correct_index)

            if not no_image:
                imgs = [np.array(img) for img in row['images_list']]
                batch_prompts_imgs.append(imgs)
            else: 
                batch_prompts_imgs = None

        assert len(batch_prompts_text) == len(dataset)
    else: 
        print
        assert len(prompts_preloads) == 3
        batch_prompts_text, batch_prompts_imgs, gts = prompts_preloads
        assert len(batch_prompts_text) == len(dataset)

    # a sense-check that the images are processed correctly
    if 0:
        _test_image_processing(batch_prompts_imgs)
    

    # call llm api
    seeds = [seed] * len(batch_prompts_text)
    print(f"Running {model} on {len(batch_prompts_text)} samples")
    if not is_rate_limited:
        responses = call_gpt_batch(texts=batch_prompts_text,
                                imgs=batch_prompts_imgs,
                                model=model,
                                num_threads=num_threads,
                                json_mode=False,
                                # overwrite_cache=True,
                                seeds=seeds)
    else:
        responses = _run_batch_chunked(batch_prompts_text, 
                           batch_prompts_imgs, seeds, model, num_threads, chunk_size=20, wait_seconds=60)

    cost = sum([c[1] for c in responses])
    msgs = [m[0] for m in responses]
    print(f"Cost of vlm call w choices: ${cost:.3f}")

    # regex the preds 
    preds = []
    for msg in msgs:
        match = re.search(REGEX_PATTERN, msg)
        if match is not None:
            pred = int(match.group(1)) - 1 # correct_index is 1-indexed but the preds are 0-indexed
            preds.append(pred)
        else:
            preds.append(-1)


    gts = np.array(gts)
    preds = np.array(preds)
    df = pd.DataFrame({
        'key_question': dataset['key_question'],
        'key_image': dataset['key_image'],
        'question': batch_prompts_text,
        'response' : msgs,
        'task': dataset['task'],
        'gt': gts,
        'pred': preds,
    })
    df['is_correct'] = (df['pred'] == df['gt'])
    df['pred_1_indexed'] = df['pred'].apply(lambda x: x + 1 if x != -1 else -1)

    if results_dir is not None:
        _save_results(df, results_dir)

    return df, (batch_prompts_text, batch_prompts_imgs, gts)


def _test_image_processing(batch_prompts_imgs):
    """
    Test that the image processing is working - needs manual inspection
    """
    batch_prompts_text = ["what is this image?"]
    batch_prompts_imgs = [batch_prompts_imgs[0]]
    responses = call_gpt_batch(texts=batch_prompts_text,
                                imgs=batch_prompts_imgs,
                                model=model,
                                num_threads=num_threads,
                                json_mode=False)
    msg = responses[0][0]
    print(msg)


def _run_batch_chunked(batch_prompts_text, batch_prompts_imgs, seeds, model, num_threads, chunk_size=20, wait_seconds=60):
    """ Some new and experimental models do per-minute rait limiting """
    responses = []
    # Add tqdm progress bar for chunks
    for i in tqdm(range(0, len(batch_prompts_text), chunk_size), desc="Processing batches"):
        chunk_texts = batch_prompts_text[i:i + chunk_size]
        chunk_imgs = batch_prompts_imgs[i:i + chunk_size] if batch_prompts_imgs else None
        chunk_seeds = seeds[i:i + chunk_size]
        
        chunk_responses = call_gpt_batch(texts=chunk_texts,
                                        imgs=chunk_imgs,
                                        model=model,
                                        num_threads=num_threads,
                                        json_mode=False,
                                        seeds=chunk_seeds)
        responses.extend(chunk_responses)
        
        if i + chunk_size < len(batch_prompts_text):  # Don't sleep after the last batch
            print(f"Sleeping for {wait_seconds} seconds for rate-limited model [{model}]")
            time.sleep(wait_seconds)

    return responses

def _save_results(df, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / 'results.csv', index=False)

    summary_results = calculate_metrics(df)
    summary_file_path = results_dir / 'summary_results.txt'
    with open(summary_file_path, 'w') as f:
        f.write(f"Mean Accuracy: {summary_results['mean_accuracy']:.3f}\n")
        f.write(f"Bad Responses: {summary_results['bad_responses_pct']:.2f}% ({summary_results['bad_responses_pct']}\n")
        f.write("Accuracy per Task:\n")
        for task, accuracy in summary_results['accuracy_per_task'].items():
            f.write(f"{task}: {accuracy:.3f}\n")

def calculate_metrics(df):
    """
    Calculate metrics for the evaluation results.
    
    Args:
        df (pd.DataFrame): The DataFrame containing 'pred' and 'gt' columns
    
    Returns: dictionary with results
    """
    valid_indices = set(range(5))  
    df_bad_responses = df[~df['pred'].isin(valid_indices)]
    bad_responses_pct = len(df_bad_responses) / len(df)
    summary_results = {
        'mean_accuracy': df['is_correct'].mean(),
        'accuracy_per_task': df.groupby('task')['is_correct'].mean().to_dict(),
        'bad_responses_pct': bad_responses_pct
    }
    return summary_results

    

if __name__ == "__main__":
    dataset = datasets.load_dataset("jmhb/microvqa")['train']
    subset = 20
    dataset = dataset.select(range(subset))
    results = eval_qa(dataset)
    print(f"Accuracy: {acc:.3f}")