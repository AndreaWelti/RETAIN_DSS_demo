# tests/test_generator.py
import numpy as np
import pandas as pd
from retain_dss.data.generator import generate_route1, generate_route2, generate_route3
from retain_dss.data.schema import MATERIAL_INPUTS, OUTPUTS_R1, OUTPUTS_R2, OUTPUTS_R3

def test_route1_shape():
    df = generate_route1(n=100, seed=0)
    assert len(df) == 100

def test_route1_output_columns():
    df = generate_route1(n=50, seed=0)
    for col in OUTPUTS_R1:
        assert col in df.columns, f"missing {col}"

def test_route1_no_nan():
    df = generate_route1(n=200, seed=1)
    assert not df.isnull().any().any()

def test_route1_pvc_purity_range():
    df = generate_route1(n=500, seed=2)
    assert df["pvc_purity"].between(60, 95).all()

def test_route2_pvc_purity_range():
    df = generate_route2(n=500, seed=3)
    assert df["pvc_purity"].between(85, 99.5).all()

def test_route3_ph_bell_curve():
    df = generate_route3(n=500, seed=10)
    df_mid_ph  = df[df["ph_level"].between(6, 8)]
    df_high_ph = df[df["ph_level"].between(10, 12)]
    assert df_mid_ph["plasticizer_recovery"].mean() > df_high_ph["plasticizer_recovery"].mean()

def test_route3_output_columns():
    df = generate_route3(n=50, seed=0)
    for col in OUTPUTS_R3:
        assert col in df.columns, f"missing {col}"

def test_all_routes_reproducible():
    assert generate_route1(n=10, seed=42).equals(generate_route1(n=10, seed=42))
    assert generate_route2(n=10, seed=42).equals(generate_route2(n=10, seed=42))
    assert generate_route3(n=10, seed=42).equals(generate_route3(n=10, seed=42))
