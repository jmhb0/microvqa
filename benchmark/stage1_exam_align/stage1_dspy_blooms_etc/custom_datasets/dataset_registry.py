#!/usr/bin/env python3
"""dataset_registry.py in src/microchat/custom_datasets.

Adapted from:
https://github.com/sanketx/AL-foundation-models/blob/main/ALFM/src/datasets/registry.py
"""

from enum import Enum


# import all dataset wrappers
from microchat.custom_datasets.dataset_wrappers import *


class DatasetType(Enum):
    """Enum of supported Datasets."""

    hotpotqa = HotPotQAWrapper()
    scieval = SciEvalWrapper()
    microchat = MicroChatWrapper()  # expecting file $DATA_ROOT/microchat.csv
    microchat_v2 = MicroChatV2Wrapper()  # expecting file $DATA_ROOT/microchat_v2.csv
    mol_bio_cell = Mol_Bio_CellWrapper()
    microbench = MicroBenchWrapper()
    blooms = BloomsWrapper()
    blooms_postbot = Blooms_PostBotWrapper()
    other_blooms = Other_BloomsWrapper()
    nbme_blooms = NBME_BloomsWrapper()
    organism_research = Organism_ResearchWrapper()
