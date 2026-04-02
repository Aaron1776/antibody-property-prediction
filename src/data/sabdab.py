"""SAbDab data loading and processing functions.

Two responsibilities:
  1. Data pipeline (run once in NB01): download from Zenodo, parse sequences,
     strip artifacts, drop anomalous entries, save sabdab_affinity.csv.
  2. Data loading (used by all downstream notebooks): load the saved CSV.

Source: Zenodo DOI 10.5281/zenodo.13120765 (Apache 2.0)
Do NOT use PyTDC -- incompatible with Colab's numpy environment.

The raw Zenodo CSV has an 'Antibody' column containing a Python list
literal (e.g. "['EVQL...', 'DIQM...']") -- parsed with ast.literal_eval.
"""

import ast
import re

import numpy as np
import pandas as pd
from pathlib import Path

_EXPECTED_ROWS = 491

# Full-IgG entries dropped during cleaning (anomalously long sequences)
_DROP_IDS = {'6d6u', '6d6t'}

SABDAB_URL = (
    'https://zenodo.org/records/13120765/files/'
    'antibody_affinity_protein_sabdab.csv?download=1'
)


# ---------------------------------------------------------------------------
# CSV loading (used by all downstream notebooks)
# ---------------------------------------------------------------------------

def load_sabdab(data_dir: Path) -> pd.DataFrame:
    """Load sabdab_affinity.csv.

    Returns a DataFrame with 491 rows. Asserts row count and that pKd column
    is present.

    Columns: Antibody_ID, heavy_seq, light_seq, Antigen_ID, Antigen, Y, pKd
    - heavy_seq, light_seq: clean amino acid strings (artifacts stripped)
    - Y: raw Kd in molar units
    - pKd = -log10(Y): training label, mean ~8.23, range 3.70-12.40
    """
    path = Path(data_dir) / 'sabdab_affinity.csv'
    df = pd.read_csv(path)

    assert len(df) == _EXPECTED_ROWS, (
        f"Expected {_EXPECTED_ROWS} rows, got {len(df)}"
    )
    assert 'pKd' in df.columns, "pKd column not found in sabdab_affinity.csv"
    return df


# ---------------------------------------------------------------------------
# Data pipeline functions (run once in NB01)
# ---------------------------------------------------------------------------

def strip_artifacts(seq: str) -> str:
    """Strip common sequence artifacts from antibody sequences.

    Handles three categories identified during EDA:
      - C-terminal tags: His-tags, flexible linkers (GGGGS), TEV sites,
        thrombin sites, FLAG tags, StrepII tags, Factor Xa sites
      - N-terminal artifacts: His-tags, TEV sites, GGGGS linkers
      - Variants of the above at different positions

    97 sequences in the SAbDab dataset had artifacts that were stripped.

    Parameters
    ----------
    seq : raw amino acid string

    Returns
    -------
    str: cleaned sequence
    """
    # --- N-terminal artifacts ---
    seq = re.sub(r'^[MGS]{0,3}H{4,}[GS]{0,3}', '', seq)  # N-term His-tag
    seq = re.sub(r'^ENLYFQ[GS]?', '', seq)                 # N-term TEV site
    seq = re.sub(r'^(GGGGS)+', '', seq)                    # N-term GGGGS linker

    # --- C-terminal artifacts ---
    seq = re.sub(r'H{6,}.*$', '', seq)       # His-tag (6+ histidines)
    seq = re.sub(r'(GGGGS)+.*$', '', seq)    # Flexible linker
    seq = re.sub(r'ENLYFQ.*$', '', seq)      # TEV site
    seq = re.sub(r'LVPRGS.*$', '', seq)      # Thrombin site
    seq = re.sub(r'SSDYKD.*$', '', seq)      # FLAG tag
    seq = re.sub(r'SHPQFEK.*$', '', seq)     # StrepII tag
    seq = re.sub(r'GIEGR.*$', '', seq)       # Factor Xa site
    seq = re.sub(r'LEHHHH.*$', '', seq)      # LEH + His-tag variant

    return seq


def parse_sabdab_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Parse and clean the raw SAbDab CSV from Zenodo.

    Applies ast.literal_eval to the 'Antibody' column to extract heavy and
    light chain sequences, computes pKd, strips artifacts, and drops the two
    full-IgG entries (6d6u, 6d6t) that are anomalously long.

    Parameters
    ----------
    raw_df : DataFrame loaded directly from the Zenodo CSV.

    Returns
    -------
    DataFrame with 491 rows and columns:
        Antibody_ID, heavy_seq, light_seq, Antigen_ID, Antigen, Y, pKd
    """
    df = raw_df.copy()

    # Parse heavy and light chains from the string-encoded list
    df['heavy_seq'] = df['Antibody'].apply(lambda x: ast.literal_eval(x)[0])
    df['light_seq'] = df['Antibody'].apply(lambda x: ast.literal_eval(x)[1])

    # Compute pKd label
    df['pKd'] = -np.log10(df['Y'])

    # Keep only the columns needed downstream
    out_cols = ['Antibody_ID', 'heavy_seq', 'light_seq', 'Antigen_ID', 'Antigen', 'Y', 'pKd']
    df = df[out_cols].copy()

    # Strip sequence artifacts
    df['heavy_seq'] = df['heavy_seq'].apply(strip_artifacts)
    df['light_seq'] = df['light_seq'].apply(strip_artifacts)

    # Drop full-IgG entries (anomalously long -- not Fab/Fv fragments)
    df = df[~df['Antibody_ID'].isin(_DROP_IDS)].copy()
    df = df.reset_index(drop=True)

    assert len(df) == _EXPECTED_ROWS, (
        f"Expected {_EXPECTED_ROWS} rows after cleaning, got {len(df)}"
    )

    return df
