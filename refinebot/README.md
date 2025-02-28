# 🤖 RefineBot
This document will guide you through the process of running our 🤖RefineBot method for removing language shortcuts on demo MCQ questions. 

## Steps:
1. **Setup.** This demo uses the OpenAI API but this can be easily replaced with any API. Export the key to your OpenAI account:
```
export OPENAI_API_KEY=<your key here>
```
3. **Data.** Format your questions like the `demo_dataset.csv` file. At minimum your dataset should include a unique question identifier (`key_question`), the question stem (`question`), a list of options (`choices`), and the index of the correct answer (`correct_index`).
4. **Customization.** Use the `demo.yaml` to specify the config. Make sure to link your dataset csv to the `run.data_file_path`. Each 🤖RefineBot element (eval, reflect, rewrite, check_rewrite) has its own prompt defined in `all_prompts.py` and specified with `prompt_key`. Each element can call its own model version, here we use a smaller cost saving model.
5. **Running the demo.** `demo.py` runs an experiment, note that seeds and number of runs per seed can be customized. `bot.py` contains the main logic for the RefineBot, which calls on `openai_api.py` to query the LLM. Results are saved using a cache.
```
python demo.py
```
6. **Output.** The final dataset will be stored as a .csv file within the experiment folder e.g. `results/<run.name>/<dataset_name>`. Each run creates a version of the questions and logs the RefineBot output at each step. The results of each run are compiled into `_samples_sorted.csv` with an overall log and statistics file.

