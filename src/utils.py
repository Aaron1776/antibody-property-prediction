"""Shared utilities used across src/ modules.

save_index / load_index are defined here to avoid duplication across the
embedding modules. Both ESM-2 and AbLang2 notebooks used identical
implementations of these helpers.
"""

import json
from pathlib import Path
from typing import Dict


def save_index(index_dict: Dict[int, str], path) -> None:
    """Save a row-index -> identifier mapping as JSON.

    Parameters
    ----------
    index_dict : {int: str}
        e.g. {0: 'Ang2_2017_G6', 1: 'EGFR_2013_Cetuximab', ...}
    path : str or Path
        Destination file path.

    Notes
    -----
    JSON requires string keys. The int keys are converted to strings on save
    and converted back to int by load_index().
    """
    str_keyed = {str(k): v for k, v in index_dict.items()}
    with open(path, 'w') as f:
        json.dump(str_keyed, f, indent=2)


def load_index(path) -> Dict[int, str]:
    """Load a JSON index file and return {int: str}.

    Reverses the string-key conversion applied by save_index().

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict with int keys and str values.
    """
    with open(path, 'r') as f:
        str_keyed = json.load(f)
    return {int(k): v for k, v in str_keyed.items()}
