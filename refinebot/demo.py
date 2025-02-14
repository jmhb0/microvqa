import os
import pandas as pd
from omegaconf import OmegaConf

from bot import run_refinebot, save_refinebot_mcqs


def demo(cfg):
    # 1. Load questions and options
    data = pd.read_csv(cfg.data_path)
    # 2. Run RefineBot on the dataset using different seeds
    for iter_seed in cfg.run.seeds:
        run_refinebot(cfg, data, iter_seed)
    # 3. Compile the results from different seeds into the final dataset
    save_refinebot_mcqs(cfg)

if __name__ == "__main__":
    exp_config_path = "demo.yaml"
    cfg = OmegaConf.load(exp_config_path)
    demo(cfg)