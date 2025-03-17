# Stage1  MCQ generation, DSPy Blooms, etc.
This folder contains a collection of scripts and utilities related to **MicroVQA**, a benchmark for multimodal reasoning in microscopy-based scientific research. This is **not a fully documented or reproducible package**, but rather a record of files to provide reference of the scripts and functions used in creation of MicroVQA. Do not post issues requesting features or submit pull-requests.

## Overview of select files
### 1. **compute_text_difficulty.py**
   - Computes readability and difficulty metrics for text-based questions.
   - Uses the `textstat` library for text analysis.
   - CLI-based script that reads an input CSV and outputs difficulty scores.

### 2. **create_examples.py**
   - Generates vector-based graphics for multiple-choice questions.
   - Uses Cairo for rendering text-based questions into images.
   - Supports metadata annotations like Bloom’s taxonomy levels.

### 3. **create_wordcloud.py**
   - Creates a word cloud from a given dataset of questions.
   - Uses `matplotlib` and `wordcloud` to generate visual representations.

### 4. **csv_to_jsonl_finetune.py**
   - Converts CSV datasets into JSONL format for fine-tuning OpenAI modelsMs.
   - Includes prompts designed to categorize questions by Bloom’s taxonomy levels.

### 5. **get_blooms_consensus.py**
   - Runs Bloom’s taxonomy classification on questions using DSPy-based models.
   - Uses retrieval models and optimization strategies to improve taxonomy predictions.

### 6. **predict_dspy.py**
   - Runs DSPy-based models for making predictions on multiple-choice questions.
   - Supports various model types, including retrieval-augmented generation.

### 7. **run_dspy.py**
   - Core script for evaluating multiple models on MicroVQA.
   - Supports different tasks (e.g., NBME-style MCQs, Bloom’s classification).
   - Interfaces with DSPy-based retrieval and optimization modules.

### 8. **mcq_metric.py**
   - Implements metrics for evaluating multiple-choice question quality.
   - Includes Bloom’s taxonomy validation and NBME-style assessment heuristics.

### 9. **base_signatures.py**
   - Defines base DSPy model signatures for question-answering and classification tasks.

### 10. **dspy_modules.py**
   - Implements DSPy-based modules for various reasoning tasks.
   - Supports retrieval-augmented generation (RAG) methods.

## Notes
- These scripts are **not** intended for direct use or deployment.
- They require dependencies including `DSPy`, `Pandas`, `Click`, `Matplotlib`, `OpenAI API`, and others.
- The folder is a record of the files but does not include installation instructions, support, or example demos.
- See the HuggingFace repository for the final, completed version of the MicroVQA benchmark.

## License and Usage
The code license can be found in the root of the repository. The code is provided **as is** for reference only. It will require refactoring and installation of all necessary dependencies for scripts to function correctly.