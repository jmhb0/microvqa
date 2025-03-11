# Fine-Tuning Bloom’s Taxonomy Classifier
This folder contains a simple script for fine-tuning **GPT-4o-mini** to classify questions based on **Bloom’s Taxonomy levels**.

## Files

- **`finetune_blooms_classifier.py`**  
  - CLI script to fine-tune an OpenAI model for Bloom’s taxonomy classification.
  - Supports uploading training and test datasets to OpenAI’s API.
  - Requires an OpenAI API key in the environment.
  
- **`blooms_classification_finetuning_train.jsonl`**  
  - Training dataset for fine-tuning.
  
- **`blooms_classification_finetuning_test.jsonl`**  
  - Test dataset for validation.

- **`checksums.txt`**  
  - File integrity checksums for verification.

## Usage

Run the fine-tuning script:

```bash
python finetune_blooms_classifier.py --train-file blooms_classification_finetuning_train.jsonl --test-file blooms_classification_finetuning_test.jsonl --upload
```

Use `--dry-run` to test execution without making API requests.

## Notes
- Requires an **OpenAI API key** environment variable (`OPENAI_API_KEY`).
- Fine-tuning costs may apply.
- This is **not a fully documented pipeline**, but a quick reference for fine-tuning experiments.
