"""
python -m ipdb refine_bot/bot.py
"""
import os
from types import SimpleNamespace
import ipdb
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import logging
import ast
from PIL import Image
import re
from pydantic import BaseModel
from omegaconf import OmegaConf
import logging
from datetime import datetime
import glob
import csv
import threading
import concurrent.futures

import all_prompts as prompts
from models.openai_api import call_gpt_batch, call_gpt

file_lock = threading.Lock()

def run_refinebot(cfg: OmegaConf,
                  df: pd.DataFrame,
                  run_number: int = 0,
                  seed: int = 0):
    """
    df must have the columns:
        question: values str
        choices: values are stringified dict with keys 
            'choices': list[str] 
            'correct_index': int
        use_case: int
    """
    # initialize logger
    dir_results, f_summary, log_str_main = config_logger(cfg, run_number, seed)

    # Prepare function args for all the questions
    func_args = []
    key_questions = df['key_question']
    for key_question in key_questions:
        row = df[df['key_question']==key_question]
        assert len(row)==1
        row = row.iloc[0]
        log_str = f"question_{key_question}"

        question_stem = row['question']
        choices = ast.literal_eval(row['choices'])
        correct_index = row['correct_index']
        use_case = row['use_case']

        if cfg.run.shuffle:
            choices, correct_index = _shuffle_choices(choices, correct_index,
                                                      seed)

        dir_log = os.path.join(dir_results, log_str)

        # Collect arguments for this question
        args = (cfg, dir_log, question_stem, choices, correct_index, f_summary,
                log_str, use_case, seed)
        func_args.append(args)

    # Run either in parallel or sequence based on flag
    if cfg.run.multiprocessing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.run.max_workers) as executor:
            futures = []
            for args in func_args:
                future = executor.submit(process_single_question, *args)
                futures.append(future)
            results = [future.result() for future in futures]

    else:
        results = []
        for args in func_args:
            result = process_single_question(*args)
            results.append(result)

    # Log summary of all results
    for log_str, iteration, use_case, return_code, cost_total in results:
        logging.info(
            f"Question {log_str}: {return_code} after {iteration} iterations. Cost: ${cost_total:.2f}"
        )
    _save_final_results(f_summary, log_str_main, cfg)

def config_logger(cfg, run_num, seed):
    exp_dir = os.path.join(cfg.run.results_dir, cfg.run.name)
    os.makedirs(exp_dir, exist_ok=True)

    # logging filename and folder name has run number
    log_str = f"run_{run_num:04d}_{seed}_{cfg.run.name}"
    log_filename = os.path.join(exp_dir, f"{log_str}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename),  # Logs everything to the file
            logging.StreamHandler(sys.stdout),  # Logs info and above to stdout
            logging.StreamHandler(
                sys.stderr)  # Logs errors and above to stderr
        ])
    logging.getLogger().handlers[2].setLevel(logging.ERROR)

    # experiment-level results directory
    dir_results = os.path.join(exp_dir, f"res_{log_str}")
    os.makedirs(dir_results, exist_ok=True)

    # save the config and dump to the logger
    f_save = os.path.join(dir_results, f"cfg.json")
    with open(f_save, 'w') as fp:
        json.dump(OmegaConf.to_container(cfg), fp, indent=4)
    logging.info("Config:")
    logging.info(json.dumps(OmegaConf.to_container(cfg), indent=4))

    # set up the results csv
    f_summary = os.path.join(exp_dir, f"sum_{log_str}_samples_async.csv")
    with file_lock:
        with open(f_summary, mode='a', newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                ["log_str", "iterations", "use_case", "code", "cost"])

    return dir_results, f_summary, log_str

def _shuffle_choices(choices, correct_index, seed_shuffle=0):
    np.random.seed(seed_shuffle)
    idxs = np.arange(len(choices))
    idxs = np.random.permutation(idxs)
    choices_shuffled = [choices[idx] for idx in idxs]
    correct_index_new = int(np.where(idxs == correct_index)[0][0])

    return choices_shuffled, correct_index_new

def process_single_question(cfg, dir_log, question_stem, choices,
                            correct_index, f_summary, log_str,
                            use_case, seed):
    """
    Processes a single question with the RefineBot and do the basic logging
    """
    result = revise_mcq(
        cfg,
        question_stem,
        choices,
        correct_index,
        dir_log,
        max_iters=cfg.run.max_iters,
        seed=seed,
        log_str=log_str,
    )

    # unpack results
    return_code, iteration, question_stem, choices, data = result

    # evals_, reflections_, rewrites_, check_rewrites_
    cost_total = _log_costs(cfg, **vars(data))

    # save results
    with open(f_summary, mode='a', newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [log_str, iteration, use_case, return_code, f"{cost_total:.2f}"])

    return log_str, iteration, use_case, return_code, cost_total

def revise_mcq(cfg: OmegaConf,
               question_stem: str,
               choices: list[str],
               correct_index: int,
               dir_log: str,
               max_iters: int = 5,
               seed: int = 0,
               log_str: str = ""):
    """
    Main loop for revising the question and choices with the RefineBot.
    Includes evaluation, reflection, rewriting, and checking.
    """
    # object to hold everything
    data = SimpleNamespace(question_stems_=[],
                           choices_=[],
                           evals_=[],
                           reflections_=[],
                           rewrites_=[],
                           check_rewrites_=[],
                           costs=[])

    # save the starting options
    question_stem_original = question_stem
    choices_original = choices
    correct_index_original = correct_index

    # logging dir
    Path(dir_log).mkdir(exist_ok=True)

    # loop 
    for iteration in range(max_iters):
        logging.info(f"[{log_str}] Running iteration {iteration}")

        # log the current question and choices
        data.choices_.append(choices)
        data.question_stems_.append(question_stem)
        _log_qa(dir_log, iteration, question_stem, choices, correct_index)

        # evaluate current question without an image
        results_eval_mcq_noimage, is_eval_is_incorrect, eval_messages = evaluate_mcq_noimage(question_stem,
                                                       choices,
                                                       correct_index,
                                                       seed=seed,
                                                       cfg_eval=cfg.eval)
        data.evals_.append(results_eval_mcq_noimage)
        _log_eval(dir_log, iteration, results_eval_mcq_noimage, correct_index)

        # if eval is incorrect, then stop
        if is_eval_is_incorrect:
            if iteration == 0:
                code = "SUCCESS_NO_CHANGE"
                logging.info(
                    f"[{log_str}] {code} MCQ already failed the image-free eval. Exiting"
                )
            else:
                code = "SUCCESS_REWRITE"
                logging.info(
                    f"[{log_str}] {code} successfully failed MCQ eval after {iteration} iterations. Exiting"
                )
            return (code, iteration, question_stem, choices, data)

        # if max evals, then quit
        if iteration == max_iters - 1:
            code = "FAIL_ITERATIONS"
            logging.info(
                f"[{log_str}] {code} Quitting after {max_iters} iterations")
            return (code, iteration, question_stem, choices, data)

        # reflect on how that was possible
        result_reflection = reflect_on_mcqnoimage_pass(
            conversation=eval_messages,
            cfg_reflect=cfg.reflect,
            seed=seed,
        )
        data.reflections_.append(result_reflection)
        _log_reflections(dir_log, iteration, result_reflection)

        # rewrite the question+distractors based on past reflections
        results_rewrite_qa = rewrite_qa(
            reflections=data.reflections_,
            cfg_rewrite=cfg.rewrite,
            question_stem_original=question_stem_original,
            choices_original=choices_original,
            correct_index_original=correct_index_original,
            seed=seed,
        )
        data.rewrites_.append(results_rewrite_qa)
        _log_rewrites(dir_log, iteration, results_rewrite_qa)

        # check that the rewrite didn't change the meaning of the qa
        question_stem_new = results_rewrite_qa['mcq_qa_new']['question_stem']
        choices_new = results_rewrite_qa['mcq_qa_new']['choices']
        correct_index_new = results_rewrite_qa['mcq_qa_new']['correct_index']
        explanation_new = results_rewrite_qa['mcq_qa_new']['explanation']

        results_check_rewrite_issame = check_rewrite_issame(
            question_stem_original,
            choices_original[correct_index_original],
            question_stem_new,
            choices_new[correct_index_new],
            seed=seed,
            cfg_check_rewrite=cfg.check_rewrite)
        data.check_rewrites_.append(results_check_rewrite_issame)
        _log_check_rewrite(dir_log, iteration, results_check_rewrite_issame)

        if not results_check_rewrite_issame['response']['is_equivalent']:
            # log and then return that it's changed, rather than try to fix it.
            code = "FAIL_REWRITE"
            logging.info(
                f"[{log_str}] {code} The rewrite prompt at iter {iteration} broke something. Exiting"
            )
            return (code, iteration, question_stem, choices, data)

        # update the current estimate
        question_stem = question_stem_new
        choices = choices_new
        correct_index = correct_index_new
        explanation = explanation_new

def evaluate_mcq_noimage(question_stem: str, choices: list[str],
                         correct_index: int, cfg_eval: OmegaConf, seed: int):
    """
    Run evaluation of the MCQ without an image. 
    """
    # "no image" prefix guidance + the standard CoT prompt + regex from MMLU-pro
    prompt_prefix = prompts.prompts_eval[cfg_eval.prompt_key]['prefix']
    prompt_suffix = prompts.prompts_eval[cfg_eval.prompt_key]['suffix']
    regex_pattern = r"answer is \(?([a-zA-Z])\)?"

    # make choices string
    choices_str = ""
    letters = list("abcdefghijk")
    for letter, choice in zip(letters, choices):
        choices_str += f"({letter}) {choice}\n"

    # compose final prompt
    prompt_text = f"{prompt_prefix}\n{question_stem}\n{prompt_suffix}\n\n{choices_str}"

    # run gpt, extract prediction
    rets = []
    num_evals = cfg_eval.multi_eval
    is_correct_cnt = 0
    is_correct_lst = []
    for eval_num in range(num_evals):

        response = call_gpt(prompt_text,
                            model=cfg_eval.model,
                            seed=seed + eval_num,
                            verbose=False)
        response_text = response[0]
        pred_letter, pred_index = _extract_mc_answer(response_text, regex_pattern)
        is_correct = (correct_index == pred_index)
        is_correct_lst.append(is_correct)
        is_correct_cnt  += is_correct

        cost = response[1]['cost'] if response[1] is not None else 0

        ret = dict(messages=response[3],
                response_text=response_text,
                is_correct=is_correct,
                pred_index=pred_index,
                cost=cost)

        rets.append(ret)
    
    # do we meet the criteria for exiting? (not the most elegant but easy to understand way)
    if num_evals == 1 and is_correct_cnt == 0: 
        is_eval_incorrect = True 
    elif num_evals > 1 and is_correct_cnt <= 1:
        is_eval_incorrect = True 
    else: 
        is_eval_incorrect = False

    # take the first instance of it being correct, and get the conversation to pass
    idxs = np.where(is_correct_lst)[0]
    if len(idxs)==0:
        idx = 0
    else: 
        idx = idxs[0]
    
    # is_correct_lst
    eval_messages = rets[idx]['messages']

    return rets, is_eval_incorrect, eval_messages

def _extract_mc_answer(text_response, regex_pattern):
    matches = re.findall(regex_pattern, text_response)

    if len(matches) > 0:
        pred_letter = matches[-1]
    else:
        pred_letter = 'None'

    letters = list("abcdefghijk")
    letters_to_idx = dict(zip(letters, range(len(letters))))
    # idx_to_letters = {v: k for k, v in letters_to_idx.items()}
    index = letters_to_idx.get(pred_letter, None)

    return pred_letter, index


def reflect_on_mcqnoimage_pass(conversation: list[dict],
                               cfg_reflect: OmegaConf, seed: int):
    prompt_text = prompts.prompts_reflect[cfg_reflect.prompt_key]
    response = call_gpt(prompt_text,
                        model=cfg_reflect.model,
                        conversation=conversation,
                        seed=seed,
                        json_mode=False,
                        verbose=False)

    cost = response[1]['cost'] if response[1] is not None else 0
    return dict(conversation=response[3], response_text=response[0], cost=cost)

def rewrite_qa(reflections: list[dict], cfg_rewrite,
               question_stem_original: str, choices_original: list[str],
               correct_index_original: int, seed: int):
    """
    Each element of 'conversation' is a question + some

    strucured_output_key:
        0: structured output in prompt text. Then use llm to parse it. Used for o1 because they don't support it

    """
    conversations = [r['conversation'] for r in reflections]
    n_conversations = len(conversations)

    prompt = prompts.prompts_rewrite[cfg_rewrite.prompt_key]
    prompt = prompt.replace("{{n_chat}}", str(n_conversations))
    prompt = prompt.replace("{{n_choices}}", str(cfg_rewrite.n_choices_target))

    prompt = prompt.replace("{{question_stem_original}}",
                            question_stem_original)
    prompt = prompt.replace("{{answer_original}}",
                            choices_original[correct_index_original])

    str_conversations = _stringify_conversations_lst(conversations)
    prompt = prompt.replace("{{conversations}}", str_conversations)

    # GPT call, enforcing the structured output optionally
    response_format = prompts.McqQA
    if cfg_rewrite.strucured_output_key == 0:
        response_unstructured = call_gpt(prompt,
                                         model=cfg_rewrite.model,
                                         seed=seed,
                                         verbose=False)
        response = _enforce_llm_response_structure(response_unstructured,
                                                   response_format, seed)
        cost = response_unstructured[1]['cost'] if response_unstructured[
            1] is not None else 0
        msg = response[0]  # structured version of the OG response.
        messages = response_unstructured[3]  # full convo from before

    elif cfg_rewrite.strucured_output_key == 1:
        response = call_gpt(prompt,
                            model=cfg_rewrite.model,
                            response_format=response_format,
                            seed=seed,
                            verbose=False)
        cost = response[1]['cost'] if response[1] is not None else 0
        msg = response[0]
        messages = response[3]

    else:
        raise NotImplementedError()

    return dict(mcq_qa_new=msg, messages=messages, cost=cost)


def check_rewrite_issame(question_stem_original: str, answer_original: str,
                         question_stem_new: str, answer_new: str, seed: str,
                         cfg_check_rewrite: OmegaConf):
    """
    After revising question, run a check that the underlying content hasn't changed
    (Note: Doing 1-indexing and not 0-indexing for simplifying the prompt)
    """
    prompt = prompts.prompt_check_rewrite[cfg_check_rewrite.prompt_key]
    prompt = prompt.replace("{{question_stem_1}}", question_stem_original)
    prompt = prompt.replace("{{answer_1}}", answer_original)
    prompt = prompt.replace("{{question_stem_2}}", question_stem_new)
    prompt = prompt.replace("{{answer_2}}", answer_new)

    response_format = prompts.PromptCheck
    if cfg_check_rewrite.strucured_output_key == 0:
        response_unstructured = call_gpt(prompt,
                                         model=cfg_check_rewrite.model,
                                         seed=seed,
                                         verbose=False)
        response = _enforce_llm_response_structure(response_unstructured,
                                                   response_format, seed)
        cost = response_unstructured[1]['cost'] if response_unstructured[
            1] is not None else 0
        msg = response[0]
        messages = response_unstructured[3]

    elif cfg_check_rewrite.strucured_output_key == 1:
        response = call_gpt(prompt,
                            model=cfg_check_rewrite.model,
                            seed=seed,
                            response_format=response_format,
                            verbose=False)
        cost = response[1]['cost'] if response[1] is not None else 0
        msg = response[0]
        messages = response[3]

    return dict(response=msg, messages=messages, cost=cost)


########## Helper function for logging and string processing ##########

def _log_qa(dir_log,
            iteration,
            question_stem,
            choices,
            correct_index,
            explanation=""):
    """ """
    # log as string
    f_save = os.path.join(dir_log, f"0_qa_iter_{iteration}.txt")
    str_log = _stringify_mcq_for_logging(question_stem, choices, correct_index)
    with open(f_save, "w") as fp:
        fp.write(str_log)

    # log as json for easier reading
    f_save = os.path.join(dir_log, f"0_5_qa_iter_{iteration}.json")
    mcq_object = prompts.McqQA(question_stem=question_stem,
                               choices=choices,
                               correct_index=correct_index,
                               explanation=explanation)
    with open(f_save, "w") as fp:
        json.dump(dict(mcq_object), fp, indent=2)


def _log_eval(dir_log, iteration, results_eval_mcq_noimage, correct_index):
    """
    By turning it into a string, we use newline, which makes logging more 
    readable.
    """
    letters = list("abcdefghijk")
    idx_to_letter = dict(zip(range(len(letters)), letters))

    for i, result_eval_mcq_noimage in enumerate(results_eval_mcq_noimage):
        msgs = result_eval_mcq_noimage['messages']
        assert len(msgs) == 2
        str_log = ""

        # prompt
        str_log += f"Prompt\n{80*'-'}\n"
        assert len(msgs[0]['content']) == 1
        str_log += msgs[0]['content'][0]['text']

        # response

        str_log += f"\n{80*'-'}\nResponse (target answer is {idx_to_letter[correct_index]})\n{80*'-'}\n"
        str_log += msgs[1]['content']

        with open(os.path.join(dir_log, f"1_eval_iter_{iteration}__run_{i}.txt"), 'w') as fp:
            fp.write(str_log)

def _log_reflections(dir_log, iteration, result_reflection):
    str_log = _stringify_conversation_pretty(result_reflection['conversation'])
    with open(os.path.join(dir_log, f"2_reflection_iter_{iteration}.txt"), 'w') as fp:
        fp.write(str_log)


def _log_rewrites(dir_log, iteration, results_check_rewrite_issame):
    str_log = _stringify_conversation_pretty(
        results_check_rewrite_issame['messages'])
    with open(os.path.join(dir_log, f"3_rewrite_iter_{iteration}.txt"), 'w') as fp:
        fp.write(str_log)


def _log_check_rewrite(dir_log, iteration, results_check_rewrite_issame):
    messages = results_check_rewrite_issame['messages']
    str_log = ""
    # propmt
    str_log += f"Prompt\n{80*'*'}\n"
    str_log += messages[0]['content'][0]['text']

    # response
    res = results_check_rewrite_issame['response']
    str_log += f"\n{80*'*'}\n"
    str_log += f"Response, is_equivalent: {res['is_equivalent']}"
    str_log += f"\n{80*'*'}\n"
    str_log += f"is_equivalent: {res['explanation']}\n\n"

    with open(os.path.join(dir_log, f"4_checkrewrite_iter_{iteration}.txt"), 'w') as fp:
        fp.write(str_log)


def _log_costs(cfg, evals_, reflections_, rewrites_, check_rewrites_,
               **kwargs):
    """
    For revising a single question, compute the final cost per stage. 
    """
    cost_eval = sum([c['cost'] for c_ in evals_ for c in c_])
    cost_reflect = sum([c['cost'] for c in reflections_])
    cost_rewrites = sum([c['cost'] for c in rewrites_])
    cost_check_rewrites = sum([c['cost'] for c in check_rewrites_])
    cost_total = cost_eval + cost_reflect + cost_rewrites + cost_check_rewrites

    return cost_total

def _stringify_mcq_for_logging(question_stem, choices, correct_index):
    """ 
    for logging, make the string where you put starts around the correct ans
          (a)   ... <wrong_answer> 
        **(b)** ... <right_answer>
          (c)   ... <wrong_answer>
    """
    str_log = question_stem + "\n"
    letters = list("abcdefghijk")
    letters_lookup = dict(zip(range(len(letters)), letters))
    letter = zip()
    for i, choice in enumerate(choices):
        letter = letters_lookup[i]
        if i == correct_index:
            str_log += f"**({letter})** {choice}\n"
        else:
            str_log += f"  ({letter})   {choice}\n"

    return str_log

def _stringify_conversation_pretty(conversation):
    """
    Pretty version of `_stringify_conversation` that shows the newlines to be 
    more readable for logging
    """
    # assert each convo turn is just text
    conv_str = ""
    for c in conversation:
        if c['role'] == 'user':
            assert len(c['content']) == 1
            conv_str += f"\n{80*'-'}\nUser\n{80*'-'}\n"
            conv_str += c['content'][0]['text']
        elif c['role'] == 'assistant':

            if type(c['content']) is str:
                conv_str += f"\n{80*'-'}\nAssistant\n{80*'-'}\n"
                conv_str += c['content']
            elif type(c['content']) is dict:
                conv_str += f"\n{80*'-'}\nAssistant (json response)\n{80*'-'}\n"
                conv_str += json.dumps(c['content'], indent=2)

        else:
            raise ValueError()

    return conv_str

def _stringify_conversations_lst(conversations):
    """
    Called by `rewrite_qa`, put the list of prior conversations into a string 
    that can be added to some llm prompt. 

    """
    str_convs = ""

    # iterate through multiple conversations
    for i in range(len(conversations)):
        str_convs += f"CONVERSATION {i+1}:\n "
        str_convs += _stringify_conversation(conversations[i])
        str_convs += "\n"

    return str_convs


def _stringify_conversation(conversation):
    """
    After processing, each 'conversation' will be represented as a list like this: 
    [ 
        {'role':'user', 'content': '....'},
        {'role':'assistant', 'content': '....'},
        {'role':'user', 'content': '....'},
        {'role':'assistant', 'content': '....'},
    ]
    However, in our implementation, the 'user' content is represented as
    {'role':'user', 'content': [{'type' : 'text', 'content' : '...'}]}

    This code also fixes that to the more standard form. There are assertions 
    to make sure thatn it is as we expect, so if the input 'conversations' 
    schema changes, then this code needs changing. 
    """
    # assert each convo turn is just text
    conv_lst = []
    for c in conversation:
        if c['role'] == 'user':
            assert len(c['content']) == 1
            conv_lst.append({
                "role": "user",
                "content": c['content'][0]['text']
            })
        elif c['role'] == 'assistant':
            assert type(c['content'])
            conv_lst.append(c)

        else:
            raise ValueError()
    str_conv = json.dumps(conv_lst, indent=2)
    return str_conv

def _enforce_llm_response_structure(response_unstructured,
                                    response_format: BaseModel,
                                    seed: int,
                                    model: str = "gpt-4o-2024-08-06"):
    prompt = prompts.prompt_enforce_structure
    prompt = prompt.replace("{{original_response}}", response_unstructured[0])
    response = call_gpt(prompt,
                        model=model,
                        response_format=response_format,
                        seed=seed,
                        verbose=False)
    return response

def _save_final_results(f_summary, log_str, cfg):
    """ 
    Final results saving. 
    In multiprocessing, the logging will be out of order, so reorder the rows.
    Also write some summary stats
    """
    df = pd.read_csv(f_summary)
    df['key_question'] = [int(n[1]) for n in df['log_str'].str.split("_")]
    df_ordered = df.sort_values("key_question")

    # reordering
    f_summary_ordered = Path(
        f_summary).parent / f"sum_{log_str}_samples_sorted.csv"
    df_ordered.to_csv(f_summary_ordered, index=False)

    # summarise
    str_log = ""
    str_log += f"API calls cost ${float(df['cost'].sum()):.2f}\n\n"
    str_log += str(df.groupby('code')['code'].count())
    str_log += "\n\n"
    str_log += str(df.groupby(['use_case', 'code'])['code'].count())
    str_log += "\n\n"
    str_log += str(df.groupby(['iterations'])['code'].count())
    str_log += "\n\n"
    str_log += str(df.groupby(['code', 'iterations'])['code'].count())
    str_log += "\n\n"
    str_log += json.dumps(OmegaConf.to_container(cfg), indent=4)
    f_summary_stats = Path(f_summary).parent / f"sum_{log_str}_stats.txt"
    with open(f_summary_stats, 'w') as fp:
        fp.write(str_log)

    logging.info(
        f"Final results in \n\t{f_summary_ordered}\n\t{f_summary_stats}")
    
def get_refinebot_mcqs(cfg, run_number=0):
    """
    Compiles MCQs for the run from the original questions or succesful rewrites by RefineBot.
    """
    dir_rewrite = os.path.join(cfg.run.results_dir, cfg.run.name)

    # Retrieve the results CSV
    results_files = glob.glob(f"{dir_rewrite}/sum_run_{run_number:04d}*_sorted*")
    if not results_files:
        raise FileNotFoundError(f"No results CSV found in {dir_rewrite} for run {run_number:04d}")
    
    df_results = pd.read_csv(results_files[0])
    df_results["question_key"] = df_results["log_str"].str.split("_").str[1]

    # Locate results directory for the run
    result_dirs = glob.glob(f"{dir_rewrite}/res_run_{run_number:04d}_*")
    if len(result_dirs) != 1:
        raise ValueError(f"Expected exactly one results directory, but found {len(result_dirs)}: {result_dirs}")
    
    dir_questions = result_dirs[0]
    question_dirs = glob.glob(f"{dir_questions}/question_*")
    
    keys_question = sorted(int(Path(d).stem.split("_")[1]) for d in question_dirs)

    # Process MCQs
    data = []
    
    for key_question in keys_question:
        row = df_results[df_results["key_question"] == key_question]
        if row.empty:
            raise ValueError(f"No matching row found in results for key_question {key_question}")
        
        row = row.iloc[0]

        # Determine the appropriate MCQ file
        question_path = f"{dir_questions}/question_{key_question}"
        if "SUCCESS" not in row["code"]:
            file_path = f"{question_path}/0_5_qa_iter_0.json"
        else:
            iter_files = glob.glob(f"{question_path}/0_5_qa_iter*")
            if not iter_files:
                raise FileNotFoundError(f"No iteration files found for question {key_question}")
            
            iter_numbers = [int(Path(f).stem.split("_")[4]) for f in iter_files]
            file_path = iter_files[np.argmax(iter_numbers)]  # Select latest iteration

        # Load MCQ data
        with open(file_path, "r") as fp:
            mcq = json.load(fp)

        # Append data to list for better performance
        data.append({
            "key_question": key_question,
            "question": mcq["question_stem"],
            "choices": mcq["choices"],
            "correct_index": mcq["correct_index"],
            "code": row["code"],
            "run_id": int(run_number),
        })

    return pd.DataFrame(data)


def save_refinebot_mcqs(cfg, run_numbers, run_seeds):
    """
    Save the results of different seeds of the RefineBot into a single dataset.
    """
    final_df = []

    # Compile the results from different seeds into a single dataset
    for run_id, run_seed in zip(run_numbers, run_seeds):
        run_df = get_refinebot_mcqs(cfg, run_number=run_id)
        run_df["seed"] = run_seed
        final_df.append(run_df)

    # Concatenate all runs into a single DataFrame
    final_df = pd.concat(final_df, ignore_index=True)

    # Save to CSV
    f_save = os.path.join(cfg.run.results_dir, cfg.run.name, f"{cfg.dataset_name}_refinebot_dataset.csv")
    final_df.to_csv(f_save, index=False)

    print("*" * 80)
    print(f"Final dataset saved to {f_save}")

    
    
