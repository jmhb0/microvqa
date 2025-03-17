"""__init__.py in src/biovlmdata."""

import os
from importlib import metadata
from pathlib import Path

from dotenv import find_dotenv
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

# set env vars from .env file
load_dotenv(find_dotenv())

__version__ = metadata.version(__package__)

PROJECT_ROOT = Path(__file__).parents[2]
LOG_DIR = PROJECT_ROOT.joinpath("logs")
LOG_DIR.mkdir(exist_ok=True)
MODULE_ROOT = Path(__file__).parent
DATA_ROOT = os.getenv("DATA_ROOT", None)
if not DATA_ROOT or not Path(DATA_ROOT).exists():
    logger.error(
        f"DATA_ROOT does not exist: {DATA_ROOT}. Please set the DATA_ROOT var in the .env file."
    )
    # raise error for unset env var
    raise FileNotFoundError(f"DATA_ROOT does not exist: {DATA_ROOT}")

VERBOSE = os.getenv("VERBOSE", False)

# Configure logger
logger.configure(
    handlers=[
        {
            "sink": RichHandler(
                markup=True,
                level="INFO",
                console=Console(width=150, color_system="auto"),
            ),
            "format": "[blue]{function}[/blue]: {message}",
        }
    ]
)
del metadata  # optional, avoids polluting the results of dir(__package__)


def get_cache_dir(name: str, create_dir: bool = False) -> Path:
    """Get the cache directory for the specified name."""
    cache_dir = os.getenv(f"{name.upper()}_CACHE_DIR")
    if cache_dir and Path(cache_dir).exists():
        cache_dir = Path(cache_dir)
    elif cache_root := os.getenv("CACHE_ROOT"):
        # in case cache_root is different from DATA_ROOT.parent
        cache_dir = Path(cache_root).joinpath(f"{name}_cache")
    elif data_root := os.getenv("DATA_ROOT"):
        cache_dir = Path(data_root).parent.joinpath(f"{name}_cache")

    if cache_dir:
        logger.debug(f"Using cache directory: {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=True) if create_dir else None
        return cache_dir
    else:
        logger.error(f"Please set the {name.upper()}_CACHE_DIR environment variable.")
        raise FileNotFoundError(
            f"{name.upper()}_CACHE_DIR environment variable not set."
        )


def setup_cache_dirs(
    feature_cache: str = None, model_cache: str = None
) -> tuple[str, str]:
    """Create cache directories if they do not exist."""
    cache_dirs = [
        Path(DATA_ROOT).joinpath("feature_cache"),
        Path(DATA_ROOT).joinpath("model_cache"),
    ]
    data_root = DATA_ROOT
    default_cache_dir = Path(data_root).parent
    feature_cache = feature_cache or get_cache_dir("feature")

    def _setup_dir(cache_dir, default_dir, arg1, arg2):
        if not cache_dir:
            result = default_dir.joinpath(arg1)
            logger.info(f"{arg2}{result}")
            result.mkdir(exist_ok=True)
            return result
        elif not Path(feature_cache).exists():
            logger.info(f"Creating cache directory: {cache_dir}")
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        return cache_dir

    feature_cache = _setup_dir(
        feature_cache,
        default_cache_dir,
        "features",
        "Using default feature cache directory: ",
    )

    model_cache = model_cache or get_cache_dir("model")
    if not model_cache:
        model_cache = default_cache_dir.joinpath("models")
        logger.info(f"Using default model cache directory: {model_cache}")
        model_cache.mkdir(exist_ok=True)

    model_cache = _setup_dir(
        model_cache,
        default_cache_dir,
        "models",
        "Using default model cache directory: ",
    )

    return feature_cache, model_cache
