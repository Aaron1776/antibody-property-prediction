import pandas as pd
from pathlib import Path

_EXPECTED_ROWS = 491


def load_sabdab(data_dir: Path) -> pd.DataFrame:
    """Load sabdab_affinity.csv.

    Returns a DataFrame with 491 rows. Asserts row count and that pKd column
    is present.

    Columns: Antibody_ID, heavy_seq, light_seq, Antigen_ID, Antigen, Y, pKd
    - heavy_seq, light_seq: clean amino acid strings (artifacts stripped)
    - Y: raw Kd in molar units
    - pKd = -log10(Y): training label, mean ~8.23, range 3.70-12.40

    Note: pKd is the label used for training. Y is retained for reference.
    """
    path = Path(data_dir) / 'sabdab_affinity.csv'
    df = pd.read_csv(path)

    assert len(df) == _EXPECTED_ROWS, (
        f"Expected {_EXPECTED_ROWS} rows, got {len(df)}"
    )
    assert 'pKd' in df.columns, "pKd column not found in sabdab_affinity.csv"

    return df
