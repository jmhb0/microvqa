"""
python -m ipdb -m eval.run_eval
"""

import ipdb
import click
from eval.eval import eval_qa
import datasets

# Define recommended models
TESTED_MODELS = [
    'gpt-4o-2024-11-20',
    'o1-2024-12-17',
    'anthropic/claude-3.5-sonnet', # openrouter api doesn't include timestamp for latest model
    'anthropic/claude-3.5-haiku',
    'google/gemini-2.0-flash-001',
    'qwen/qwen-2-vl-72b-instruct',
    
]
TESTED_MODELS_RATE_LIMITED = [
    'anthropic/claude-3.7-sonnet', # openrouter api doesn't include timestamp for latest model
    'google/gemini-2.0-pro-exp-02-05:free',
    'qwen/qwen2.5-vl-72b-instruct:free',
]

@click.command()
@click.option('--subset', type=int, default=0, help='Subset size (0 for full dataset)')
@click.option('--model', type=str, default=TESTED_MODELS[0],
              help='Model to use for evaluation. Recommended models: ' + ', '.join(TESTED_MODELS))
@click.option('--no_image', is_flag=True, default=False, help='Run evaluation without images')
@click.option('--seed', type=int, default=0, help='Random seed for reproducibility')
@click.option('--num_threads', type=int, default=64, help='Number of threads for evaluation')
def main(subset, model, no_image, seed, num_threads):
    dataset = datasets.load_dataset("jmhb/microvqa")['train']
    if subset > 0:
        dataset = dataset.select(range(subset))
    
    print(f"Evaluating [{model}] on {len(dataset)} examples with no_image=[{no_image}]")
    results_dir = f"eval/results/{model}_subset_{subset}_seed_{seed}"
    is_rate_limited = model in TESTED_MODELS_RATE_LIMITED
    df = eval_qa(dataset, 
                model=model, 
                no_image=no_image, 
                seed=seed, 
                num_threads=num_threads, 
                results_dir=results_dir,
                is_rate_limited=is_rate_limited)
    print(f"Accuracy: {df['is_correct'].mean():.3f}")
    ipdb.set_trace()
    pass

def run_model_evaluation(models=TESTED_MODELS, subset=0, no_image=False, seed=0, num_threads=64):
    """
    Run evaluation for multiple models in sequence.
    
    Args:
        models (list): List of models to evaluate
        subset (int): Subset size (0 for full dataset)
        no_image (bool): Whether to run evaluation without images
        seed (int): Random seed for reproducibility
        num_threads (int): Number of threads for evaluation
    """
    results = {}
    for model in models:
        print(f"\nEvaluating model: {model}")
        dataset = datasets.load_dataset("jmhb/microvqa")['train']
        if subset > 0:
            dataset = dataset.select(range(subset))
        
        print(f"Evaluating [{model}] on {len(dataset)} examples with no_image=[{no_image}]")
        results_dir = f"eval/results/{model}_subset_{subset}_seed_{seed}"
        preds, gts, msgs, acc = eval_qa(dataset, model=model, no_image=no_image, seed=seed, num_threads=num_threads, results_dir=results_dir)
        print(f"Accuracy: {acc:.3f}")
        results[model] = {'accuracy': acc, 'predictions': preds, 'ground_truth': gts, 'messages': msgs}
    
    return results

if __name__ == "__main__":
    main()

    # run all models
    if 0:
        subset=100
        results = run_model_evaluation(
            models=TESTED_MODELS,
            subset=subset,
            no_image=False
        )
    ipdb.set_trace()
    pass

