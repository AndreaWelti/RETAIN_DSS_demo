# tests/test_loader.py
import pandas as pd
from retain_dss.data.loader import save_route, load_route
from retain_dss.data.generator import generate_route1

def test_round_trip(tmp_path):
    df = generate_route1(n=50, seed=0)
    save_route(df, "route1_mechanical", data_dir=tmp_path)
    loaded = load_route("route1_mechanical", data_dir=tmp_path)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), loaded.reset_index(drop=True))

def test_save_creates_file(tmp_path):
    df = generate_route1(n=10, seed=0)
    path = save_route(df, "route1_mechanical", data_dir=tmp_path)
    assert path.exists()
