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

REGEX_PATTERN = r"answer is \(?([0-9])\)?"

def eval_qa(dataset,
            model='gpt-4o-2024-11-20',
            no_image=False,
            seed=0,
            results_dir=None,
            num_threads=64,
            is_rate_limited=False):
    """ 
	Run eval.
	"""

    batch_prompts_text = []
    batch_prompts_imgs = []
    gts = []

    for i, row in enumerate(dataset):
        if no_image:
            prompt = PROMPT_TEMPLATE_NO_IMAGE
        else:
            prompt = PROMPT_TEMPLATE
        prompt = prompt.replace("{{QUESTION}}", row['question'])
        choices_str = ""
        for j, ch in enumerate(row['choices']):
            choices_str += f"  ({j+1}): {ch}\n" # base-1 indexing
        prompt = prompt.replace("{{CHOICES}}", choices_str)
        batch_prompts_text.append(prompt)
        gts.append(row['correct_index'])

        imgs = [np.array(img) for img in row['images_list']]
        batch_prompts_imgs.append(imgs)

    assert len(batch_prompts_text) == len(dataset)

    # a sense-check that the images are processed correctly
    if 0:
        _test_image_processing(batch_prompts_imgs)

    # call llm api
    seeds = [seed] * len(batch_prompts_text)
    if no_image:
        batch_prompts_imgs = None
    print(f"Running {model} on {len(batch_prompts_text)} samples")
    if not is_rate_limited:
        responses = call_gpt_batch(texts=batch_prompts_text,
                                imgs=batch_prompts_imgs,
                                model=model,
                                num_threads=num_threads,
                                json_mode=False,
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
            pred = int(match.group(1)) - 1
            preds.append(pred)
        else:
            preds.append(-1)

    # save response
    ipdb.set_trace()

    gts = np.array(gts)
    preds = np.array(preds)
    df = pd.DataFrame({
        'question': batch_prompts_text,
        'response' : msgs,
        'task': dataset['task'],
        'gt': gts,
        'pred': preds,
    })
    df['is_correct'] = (df['pred'] == df['gt'])

    if results_dir is not None:
        _save_results(df, results_dir)

    return df


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
    ipdb.set_trace()
    pass

def _run_batch_chunked(batch_prompts_text, batch_prompts_imgs, seeds, model, num_threads, chunk_size=20, wait_seconds=60):
    """ Some new and experimental models do per-minute rait limiting """
    responses = []
    for i in range(0, len(batch_prompts_text), chunk_size):
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
            print(f"Sleeping for {wait_seconds} seconds for rate-limited model [{model}df]")
            time.sleep(wait_seconds)

    return responses

def _save_results(df, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / 'results.csv', index=False)
    summary_results = {
        'mean_accuracy': df['is_correct'].mean(),
        'accuracy_per_task': df.groupby('task')['is_correct'].mean().to_dict()
    }

    summary_file_path = results_dir / 'summary_results.txt'
    with open(summary_file_path, 'w') as f:
        f.write(f"Mean Accuracy: {summary_results['mean_accuracy']:.3f}\n")
        f.write("Accuracy per Task:\n")
        for task, accuracy in summary_results['accuracy_per_task'].items():
            f.write(f"{task}: {accuracy:.3f}\n")

    

if __name__ == "__main__":
    dataset = datasets.load_dataset("jmhb/microvqa")['train']
    subset = 20
    dataset = dataset.select(range(subset))
    preds, gts, msgs, acc = eval_qa(dataset)
    print(f"Accuracy: {acc:.3f}")
    ipdb.set_trace()
    pass