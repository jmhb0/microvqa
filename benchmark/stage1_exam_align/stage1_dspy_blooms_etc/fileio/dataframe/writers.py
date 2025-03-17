#!/usr/bin/env python3
"""writer.py in src/microchat/fileio/dataframe."""

from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger

try:
    from pyarrow import feather

    use_feather = False
except ModuleNotFoundError:
    logger.warning("pyarrow not found. Using pandas instead.")
    use_feather = False


def save_df_to_file(df: pd.DataFrame, output_file: Union[Path, str]) -> None:
    """Save DataFrame to file."""
    if output_file.suffix == ".csv":
        df.to_csv(output_file, index=False)
    elif output_file.suffix == ".parquet":
        df.to_parquet(output_file)
    elif output_file.suffix == ".feather" and use_feather:
        feather.write_feather(df, output_file.as_posix())
    else:
        logger.error(f"Unsupported file format: {output_file.suffix}")
        raise ValueError(f"Unsupported file format: {output_file.suffix}")

    logger.info(f"Saved DataFrame to {output_file}")
