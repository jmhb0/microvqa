"""
python -m ipdb -m eval.run_eval
"""

import ipdb
import click
import pandas as pd
from eval.eval import eval_qa, calculate_metrics
import datasets

TESTED_MODELS = [
    'gpt-4o-2024-11-20',
    'gpt-4o-2024-08-06',   # the model used by refinebot
    'gpt-4o-mini-2024-07-18', # the closest mini equivalent to gpt-4o-2024-08-06, which was used by the refinebot
    'o1-2024-12-17',
    'anthropic/claude-3.5-sonnet', # openrouter api doesn't include timestamp for latest model
    'anthropic/claude-3.5-sonnet-20240620', # I *think* its the sonnet model used by refinebot
    'anthropic/claude-3.5-haiku',
    'google/gemini-pro-1.5',
    'google/gemini-2.0-flash-001',
    'google/gemini-flash-1.5-8b',
    'qwen/qwen-2-vl-72b-instruct',
    'qwen/qwen-2-vl-7b-instruct',
    'meta-llama/llama-3.2-90b-vision-instruct',
    'meta-llama/llama-3.2-11b-vision-instruct',
    'nvidia/llama-3.1-nemotron-70b-instruct',
    'x-ai/grok-2-vision-1212',
    'mistralai/pixtral-large-2411',
    'mistralai/pixtral-12b'
]

# 'qwen/qwen2.5-vl-72b-instruct', # ? weird error 
TESTED_MODELS_RATE_LIMITED = [
    ## per day limit
    'google/gemini-2.0-pro-exp-02-05:free',
    'google/gemma-3-27b-it:free',
    ## per minute limit only (I think)
    # 'anthropic/claude-3.7-sonnet', # openrouter api doesn't include timestamp for latest model
    'qwen/qwen2.5-vl-72b-instruct:free',
]

MODELS_STAGE1 = [
    'gpt-4o-2024-11-20',
    'gpt-4o-mini-2024-07-18',
    'anthropic/claude-3.5-sonnet', # openrouter api doesn't include timestamp for latest model
    'anthropic/claude-3.5-haiku',
    # random other models
    'mistralai/pixtral-large-2411',
    'google/gemini-pro-1.5',
    'qwen/qwen-2-vl-72b-instruct',
    'google/gemini-2.0-flash-001',
    'o1-2024-12-17',
]


TEST_MODEL_ABLATIONS = [
    'o1-2024-12-17',
    'gpt-4o-2024-11-20',
    'gpt-4o-mini-2024-07-18', # the closest mini equivalent to gpt-4o-2024-08-06, which was used by the refinebot
    'anthropic/claude-3.5-sonnet', # openrouter api doesn't include timestamp for latest model
    'anthropic/claude-3.5-haiku',
    'google/gemini-pro-1.5',
    'mistralai/pixtral-large-2411',
    'x-ai/grok-2-vision-1212',
    'mistralai/pixtral-12b',
]

@click.command()
@click.option('--subset', type=int, default=0, help='Subset size (0 for full dataset)')
@click.option('--model', type=str, default=TESTED_MODELS[0],
              help='Model to use for evaluation. Recommended models: ' + ', '.join(TESTED_MODELS))
@click.option('--no_image', type=click.BOOL, default=False, help='Run evaluation without images')
@click.option('--seed', type=int, default=0, help='Random seed for reproducibility')
@click.option('--num_threads', type=int, default=64, help='Number of threads for evaluation')
@click.option('--is_stage1', type=click.BOOL, default=False, help='Run evaluation for stage 1')
def main(subset, model, no_image, seed, num_threads, is_stage1):
    dataset = datasets.load_dataset("jmhb/microvqa")['train']
    if subset > 0:
        dataset = dataset.select(range(subset))
    
    print(f"Evaluating [{model}] on {len(dataset)} examples with no_image=[{no_image}] is_stage1=[{is_stage1}]")
    results_dir = f"eval/results/{model.replace('/', '_')}_noimage_{no_image}_isstage1_{is_stage1}_subset_{subset}_seed_{seed}"
    print(f"Results will be saved in {results_dir}")

    is_rate_limited = model in TESTED_MODELS_RATE_LIMITED
    df, _ = eval_qa(dataset, 
                model=model, 
                no_image=no_image, 
                seed=seed, 
                num_threads=num_threads, 
                results_dir=results_dir,
                is_stage1=is_stage1,
                is_rate_limited=is_rate_limited)
    print(f"Accuracy: {df['is_correct'].mean():.3f}")
    return df
    



def run_multiple_models_evaluation(models=TESTED_MODELS, 
                                   subset=0, 
                                   no_image=False, 
                                   seed=0, 
                                   num_threads=64, 
                                   do_summarize_only=False, 
                                   is_noimage_v2=False,
                                   is_noimage_v3=False,
                                   is_stage1=False):
    """
    Run evaluation for multiple models in sequence.
    
    Args:
        models (list): List of models to evaluate
        subset (int): Subset size (0 for full dataset)
        no_image (bool): Whether to run evaluation without images
        seed (int): Random seed for reproducibility
        num_threads (int): Number of threads for evaluation
    """
    if do_summarize_only:
        return summarize_only(models, subset, no_image=no_image, is_stage1=is_stage1, seed=seed, is_noimage_v2=is_noimage_v2, is_noimage_v3=is_noimage_v3)

    results = {}
    prompts_preloads = None
    for model in models:
        print(f"\nEvaluating model: {model}")
        dataset = datasets.load_dataset("jmhb/microvqa")['train']
        if subset > 0:
            dataset = dataset.select(range(subset))
        
        
        print(f"Evaluating [{model}] on {len(dataset)} examples with no_image=[{no_image}]")
        results_dir = f"eval/results/{model.replace('/', '_')}_noimage_{no_image}_isstage1_{is_stage1}_subset_{subset}_seed_{seed}"
        if is_noimage_v2:
            results_dir += "_noimagev2"
        if is_noimage_v3:
            results_dir += "_noimagev3"
        is_rate_limited = model in TESTED_MODELS_RATE_LIMITED
        # try:
        df, prompts_preloads = eval_qa(dataset, 
                                       model=model, 
                                       no_image=no_image, 
                                       seed=seed, 
                                       num_threads=num_threads, 
                                       results_dir=results_dir, 
                                       prompts_preloads=prompts_preloads, 
                                       is_stage1=is_stage1,
                                       is_noimage_v2=is_noimage_v2,
                                       is_noimage_v3=is_noimage_v3,
                                       is_rate_limited=is_rate_limited)
        ipdb.set_trace()
        
        print(f"Accuracy: {df['is_correct'].mean():.3f}")
        results[model] = {'accuracy': df['is_correct'].mean()}#'predictions': preds, 'ground_truth': gts, 'messages': msgs}
        # except Exception as e:
        #     print(f"error for model {model}")
        #     print(str(e))
    
    return results

def summarize_only(models, subset, no_image=False, is_stage1=False, seed=0, is_noimage_v2=False, is_noimage_v3=False):
    results = {}
    rows = []
    for model in models:
        results_dir = f"eval/results/{model.replace('/', '_')}_noimage_{no_image}_isstage1_{is_stage1}_subset_{subset}_seed_{seed}"
        print(results_dir)
        if is_noimage_v2:
            results_dir += "_noimagev2"
        if is_noimage_v3:
            results_dir += "_noimagev3"
        df = pd.read_csv(f"{results_dir}/results.csv")
        res = calculate_metrics(df)
        results[model] = res

        rows.append([
            model, 
            res['mean_accuracy'], 
            res['accuracy_per_task'][1], 
            res['accuracy_per_task'][2], 
            res['accuracy_per_task'][3], 
            res['bad_responses_pct']
        ])

    df_summary = pd.DataFrame(rows, columns=['model', 'acc', 'acc_task1', 'acc_task2', 'acc_task3', 'bad_response'])
    fname = f"eval/results/summary_results_{subset}_seed_{seed}_noimage_{no_image}_isstage1_{is_stage1}_noimagev2_{is_noimage_v2}_noimagev3_{is_noimage_v3}.csv"
    df_summary.to_csv(fname, index=False)
    print(f"Summary results saved to {fname}")

    return df_summary

    

if __name__ == "__main__":
    main()

    ## run all models
    if 0:
        subset=0
        do_summarize_only = False
        models = TESTED_MODELS
        # models = TESTED_MODELS_RATE_LIMITED
        # models = MODELS_STAGE1
        # models = TEST_MODEL_ABLATIONS
        
        is_stage1 = False
        no_image = False
        is_noimage_v2 = False # no image or question text, only choices
        is_noimage_v3 = False # no image and only the last sentence of the question, which removes context

        df_results = run_multiple_models_evaluation(
                models=models,
                subset=subset,
                no_image=no_image,
                do_summarize_only=do_summarize_only,
                is_stage1=is_stage1,
                is_noimage_v2=is_noimage_v2,
                is_noimage_v3=is_noimage_v3,
            )
    ipdb.set_trace()
    pass

