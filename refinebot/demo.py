import os
import glob
import pandas as pd
from omegaconf import OmegaConf

from bot import run_refinebot, save_refinebot_mcqs


def demo(cfg):
    # 0. Make sure experiment directory is empty for compiling the results, cache will deal with partial runs
    if os.path.exists(os.path.join(cfg.run.results_dir, cfg.run.name)):
        print(f"Removing existing directory: {os.path.join(cfg.run.results_dir, cfg.run.name)}")
        os.system(f"rm -r {os.path.join(cfg.run.results_dir, cfg.run.name)}")
    # 1. Load questions and options
    data = pd.read_csv(cfg.run.data_file_path)
    # 2. Run RefineBot on the dataset using different seeds
    # making run number same as seed but the experiment can be run multiple times with the same seed
    run_numbers = range(len(cfg.run.seeds))
    for run_number, iter_seed in zip(run_numbers, cfg.run.seeds):
        sorted_files = glob.glob(os.path.join(cfg.run.results_dir, cfg.run.name, f"*{run_number}*_sorted.csv"))
        if len(sorted_files) > 0:
            print(f"Skipping run {run_number} as it already exists")
            continue
        run_refinebot(cfg, data, run_number, iter_seed)
    # 3. Compile the results from different seeds into the final dataset
    save_refinebot_mcqs(cfg, run_numbers, cfg.run.seeds)

if __name__ == "__main__":
    exp_config_path = "demo.yaml"
    cfg = OmegaConf.load(exp_config_path)
    demo(cfg)