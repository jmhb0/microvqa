# Evaluation
## Accessing the dataset 
```
from datasets import load_dataset
dataset = load_dataset("jmhb/microvqa")['train']
```
See the dataset details in [huggingface.co/datasets/jmhb/microvqa](https://huggingface.co/datasets/jmhb/microvqa). The tldr is that each row has a question, a list of images, a list of multiple-choice options, and an index for the correct answer. 


## Running evaluation via API
Run for one model:
```
python -m eval.run_eval --model gpt-4o-2024-11-20 --subset 0 --no_image False --seed 0 
```
- model (default `'gpt-4o-2024-11-20'`): any model compatible with openai API from [these docs](https://platform.openai.com/docs/models/) (set environment variable `OPENAI_API_KEY`) or OpenRouter APIs from [these docs](https://openrouter.ai/) (set environment variable `OPENROUTER_API_KEY`). See [run_eval.py](run_eval.py) for a list of models we tested that will work with this script.
- subset (default 0): 0 for full dataset, otherwise `N` for first N examples.
- no_image (default False): whether to run the evaluation without images.
- seed (default 0): random seed.

Predictions are saved in `eval/results/{model}_subset_{subset}_seed_{seed}/`, including a csv of LLM responses and a text file of summary stats.

The llava and llava-med models were run in a separate codebase: (TODO).

