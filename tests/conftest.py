# tests/conftest.py
"""
Pytest fixtures для тестов.
"""

import sys
from pathlib import Path

# === ВАЖНО: добавляем корень проекта в sys.path ===
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_tabular_df():
    """Пример табличных данных"""
    np.random.seed(42)
    n = 200
    
    df = pd.DataFrame({
        "numeric1": np.random.randn(n),
        "numeric2": np.random.randn(n) * 10 + 5,
        "numeric3": np.random.exponential(2, n),
        "category1": np.random.choice(["A", "B", "C"], n),
        "category2": np.random.choice(["X", "Y"], n),
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3])
    })
    
    # Добавляем пропуски
    df.loc[0:10, "numeric1"] = np.nan
    df.loc[5:15, "category1"] = np.nan
    
    return df


@pytest.fixture
def sample_text_df():
    """Пример текстовых данных"""
    texts = [
        "This is a great product, I love it!",
        "Terrible experience, would not recommend.",
        "Average quality, nothing special.",
        "Best purchase ever, highly recommended!",
        "Waste of money, very disappointed.",
    ] * 40
    
    labels = [1, 0, 1, 1, 0] * 40
    
    return pd.DataFrame({
        "review": texts,
        "sentiment": labels
    })


@pytest.fixture
def sample_imbalanced_df():
    """Пример сильно несбалансированных данных"""
    np.random.seed(42)
    
    return pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "target": [0] * 90 + [1] * 10
    })


@pytest.fixture
def tmp_image_dir(tmp_path):
    """Временная директория с изображениями"""
    try:
        import cv2
    except ImportError:
        pytest.skip("opencv not installed")
    
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    # Создаём несколько тестовых изображений
    for i in range(10):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i}.jpg"), img)
    
    return img_dir


# === Дополнительные фикстуры ===

@pytest.fixture
def simple_df():
    """Простой DataFrame без пропусков"""
    np.random.seed(42)
    return pd.DataFrame({
        'numeric1': np.random.randn(100),
        'numeric2': np.random.randn(100) * 10 + 5,
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })


@pytest.fixture  
def sample_tabular_data(sample_tabular_df):
    """Алиас для совместимости с test_forge.py"""
    return sample_tabular_df


@pytest.fixture
def sample_text_data(sample_text_df):
    """Алиас для совместимости с test_forge.py"""
    return sample_text_df