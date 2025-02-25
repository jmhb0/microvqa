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
- model: any model compatible with openai API from [these docs](https://platform.openai.com/docs/models/) (set environment variable `OPENAI_API_KEY`) or OpenRouter APIs from [these docs](https://openrouter.ai/) (set environment variable `OPENROUTER_API_KEY`). Default is `gpt-4o-2024-11-20`. See [run_eval.py](run_eval.py) for a list of models we tested that will work with this script.
- subset: 0 for full dataset, otherwise subset=N for first N examples. Default is 0.
- no_image: whether to run the evaluation without images. Default is False.
- seed: random seed.

Predictions are saved in `eval/results/{model}_subset_{subset}_seed_{seed}/`, including a csv of LLM responses and a text file of summary stats.

The llava and llava-med models were run in 
- llava and llava-med: 

