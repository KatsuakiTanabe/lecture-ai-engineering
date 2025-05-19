import os
import pytest
import pandas as pd
import numpy as np
import great_expectations as gx
from sklearn.datasets import fetch_openml
import warnings

# 警告を抑制
warnings.filterwarnings("ignore")

# テスト用データパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")


@pytest.fixture
def sample_data():
    """Titanicテスト用データセットを読み込む"""
    return pd.read_csv(DATA_PATH)


def test_data_volume(sample_data):
    """データ件数が一定以上あることを確認"""
    assert len(sample_data) >= 300, f"データ件数が少なすぎます: {len(sample_data)} 件（最低300件必要）"