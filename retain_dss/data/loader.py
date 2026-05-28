# retain_dss/data/loader.py
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "synthetic"


def save_route(df: pd.DataFrame, route_name: str, data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{route_name}.csv"
    df.to_csv(path, index=False)
    return path


def load_route(route_name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / f"{route_name}.csv"
    return pd.read_csv(path)
