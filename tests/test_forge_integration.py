# tests/test_forge_integration.py
"""
Интеграционные тесты для AutoForge.

Тестирует полный пайплайн без моков.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from automl_data.core.forge import AutoForge, ForgeResult
from automl_data.core.container import DataContainer
from automl_data.utils.exceptions import ValidationError


# ==================== FIXTURES ====================

@pytest.fixture
def complete_test_df() -> pd.DataFrame:
    """Полный тестовый датасет со всеми типами данных"""
    np.random.seed(42)
    n = 200
    
    df = pd.DataFrame({
        'num1': np.random.randn(n),
        'num2': np.random.uniform(0, 100, n),
        'cat1': np.random.choice(['A', 'B', 'C', 'D'], n),
        'cat2': np.random.choice(['X', 'Y'], n),
        'target': np.random.choice([0, 1], n)
    })
    
    # Добавляем пропуски
    df.loc[np.random.choice(n, 20, replace=False), 'num1'] = np.nan
    df.loc[np.random.choice(n, 15, replace=False), 'num2'] = np.nan
    
    return df


@pytest.fixture
def imbalanced_test_df() -> pd.DataFrame:
    """Несбалансированный датасет"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.uniform(0, 100, 100),
        'category': np.random.choice(['A', 'B'], 100),
        'target': [0] * 85 + [1] * 15  # 85/15 дисбаланс
    })


@pytest.fixture
def regression_test_df() -> pd.DataFrame:
    """Датасет для регрессии"""
    np.random.seed(42)
    n = 150
    return pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.uniform(0, 100, n),
        'feature3': np.random.randint(0, 50, n),
        'target': np.random.uniform(100, 10000, n)  # Непрерывная переменная
    })


@pytest.fixture
def multiclass_test_df() -> pd.DataFrame:
    """Мультиклассовый датасет"""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.uniform(0, 100, n),
        'target': np.random.choice([0, 1, 2, 3], n)
    })


# ==================== COMPLETE PIPELINE TESTS ====================

class TestAutoForgeIntegration:
    """Интеграционные тесты AutoForge"""
    
    def test_complete_tabular_pipeline(self, complete_test_df):
        """Полный табличный пайплайн без моков"""
        forge = AutoForge(
            target="target",
            task="classification",
            impute_strategy="simple",
            scaling="standard",
            encode_strategy="auto",
            balance=True,
            verbose=False
        )
        
        result = forge.fit_transform(complete_test_df)
        
        # Проверяем результат
        assert isinstance(result, ForgeResult)
        assert result.data is not None
        assert len(result.data) > 0
        
        # Пропуски должны быть заполнены
        assert result.data.isnull().sum().sum() == 0
        
        # Target должен сохраниться
        assert 'target' in result.data.columns
        assert result.y is not None
        
        # Quality score должен быть валидным
        assert 0.0 <= result.quality_score <= 1.0
        
        # Должны быть выполнены шаги
        assert len(result.steps) > 0
        
        print(f"\n✅ Полный пайплайн:")
        print(f"   Исходный размер: {complete_test_df.shape}")
        print(f"   Результат: {result.shape}")
        print(f"   Качество: {result.quality_score:.0%}")
        print(f"   Шаги: {result.steps}")
    
    def test_pipeline_with_different_strategies(self, complete_test_df):
        """Тест с разными стратегиями"""
        strategies = [
            {"impute_strategy": "simple", "scaling": "standard"},
            {"impute_strategy": "knn", "scaling": "robust"},
            {"impute_strategy": "simple", "scaling": "minmax"},
        ]
        
        for strategy in strategies:
            forge = AutoForge(
                target="target",
                task="classification",
                balance=False,  # Отключаем балансировку для ускорения
                verbose=False,
                **strategy
            )
            
            result = forge.fit_transform(complete_test_df)
            
            assert result.data is not None
            assert result.data.isnull().sum().sum() == 0
            
            print(f"✅ Стратегия {strategy}: OK")
    
    def test_classification_with_balancing(self, imbalanced_test_df):
        """Тест классификации с балансировкой"""
        original_distribution = imbalanced_test_df['target'].value_counts().to_dict()
        
        forge = AutoForge(
            target="target",
            task="classification",
            balance=True,
            balance_threshold=0.3,
            verbose=False
        )
        
        result = forge.fit_transform(imbalanced_test_df)
        
        new_distribution = result.data['target'].value_counts().to_dict()
        
        # Минорный класс должен увеличиться
        assert new_distribution.get(1, 0) >= original_distribution[1]
        
        print(f"\n✅ Балансировка:")
        print(f"   До: {original_distribution}")
        print(f"   После: {new_distribution}")
    
    def test_regression_pipeline(self, regression_test_df):
        """Тест регрессионного пайплайна"""
        forge = AutoForge(
            target="target",
            task="regression",
            balance=False,  # Для регрессии не нужно
            verbose=False
        )
        
        result = forge.fit_transform(regression_test_df)
        
        assert result.data is not None
        assert result.y is not None
        
        # Target должен быть непрерывным
        assert result.y.nunique() > 20
        
        # Проверяем, что размер не изменился (нет балансировки)
        assert len(result.data) == len(regression_test_df)
    
    def test_multiclass_classification(self, multiclass_test_df):
        """Тест мультиклассовой классификации"""
        forge = AutoForge(
            target="target",
            task="classification",
            balance=True,
            verbose=False
        )
        
        result = forge.fit_transform(multiclass_test_df)
        
        # Все классы должны сохраниться
        original_classes = set(multiclass_test_df['target'].unique())
        new_classes = set(result.data['target'].unique())
        
        assert original_classes == new_classes
    
    def test_quality_score_calculation(self, complete_test_df):
        """Тест расчёта quality score"""
        forge = AutoForge(
            target="target",
            verbose=False
        )
        
        result = forge.fit_transform(complete_test_df)
        
        # Quality score должен быть в диапазоне [0, 1]
        assert 0.0 <= result.quality_score <= 1.0
        
        # После обработки качество должно быть достаточно высоким
        assert result.quality_score > 0.5
    
    def test_get_splits(self, complete_test_df):
        """Тест получения сплитов"""
        forge = AutoForge(
            target="target",
            test_size=0.2,
            verbose=False
        )
        
        result = forge.fit_transform(complete_test_df)
        
        X_train, X_test, y_train, y_test = result.get_splits()
        
        # Проверяем размеры
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        
        assert 0.15 < test_ratio < 0.25
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        
        # X не должен содержать target
        assert 'target' not in X_train.columns
        assert 'target' not in X_test.columns
    
    def test_get_splits_custom_params(self, complete_test_df):
        """Тест сплитов с кастомными параметрами"""
        forge = AutoForge(target="target", verbose=False)
        result = forge.fit_transform(complete_test_df)
        
        X_train, X_test, y_train, y_test = result.get_splits(
            test_size=0.3,
            random_state=123
        )
        
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        
        assert 0.25 < test_ratio < 0.35
    
    def test_reproducibility(self, complete_test_df):
        """Тест воспроизводимости с random_state"""
        forge1 = AutoForge(
            target="target",
            random_state=42,
            verbose=False
        )
        
        forge2 = AutoForge(
            target="target",
            random_state=42,
            verbose=False
        )
        
        result1 = forge1.fit_transform(complete_test_df.copy())
        result2 = forge2.fit_transform(complete_test_df.copy())
        
        # Результаты должны быть одинаковыми
        assert result1.shape == result2.shape
        
        # Сплиты должны быть одинаковыми
        X1_train, X1_test, _, _ = result1.get_splits(random_state=42)
        X2_train, X2_test, _, _ = result2.get_splits(random_state=42)
        
        assert len(X1_train) == len(X2_train)
        assert len(X1_test) == len(X2_test)
    
    def test_to_numpy(self, complete_test_df):
        """Тест конвертации в numpy"""
        forge = AutoForge(target="target", verbose=False)
        result = forge.fit_transform(complete_test_df)
        
        X, y = result.to_numpy()
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == len(y)
    
    def test_summary(self, complete_test_df):
        """Тест получения summary"""
        forge = AutoForge(target="target", verbose=False)
        result = forge.fit_transform(complete_test_df)
        
        summary = result.summary()
        
        assert isinstance(summary, dict)
        assert 'execution_time' in summary
        assert 'steps' in summary
        assert 'config' in summary
    
# ==================== ERROR HANDLING TESTS ====================

class TestErrorHandling:
    """Тесты обработки ошибок"""
    
    def test_invalid_target_column(self, complete_test_df):
        """Тест с невалидной целевой колонкой"""
        forge = AutoForge(
            target="nonexistent_column",
            verbose=False
        )
        
        with pytest.raises(ValidationError):
            forge.fit_transform(complete_test_df)
    
    def test_empty_dataframe(self):
        """Тест с пустым DataFrame"""
        empty_df = pd.DataFrame(columns=['a', 'b', 'target'])
        
        forge = AutoForge(target="target", verbose=False)
        
        with pytest.raises(ValidationError):
            forge.fit_transform(empty_df)
    
    def test_transform_without_fit(self, complete_test_df):
        """Тест transform без fit"""
        forge = AutoForge(target="target", verbose=False)
        
        with pytest.raises((RuntimeError, Exception)) as exc_info:
            forge.transform(complete_test_df)
        
        # Должно быть сообщение о необходимости fit
        assert "fit" in str(exc_info.value).lower()
    
    def test_missing_target_values(self):
        """Тест с пропусками в target"""
        df = pd.DataFrame({
            'feature': np.random.randn(100),
            'target': [0, 1, np.nan] * 33 + [0]
        })
        
        forge = AutoForge(target="target", verbose=False)
        
        # Должно обработаться без падения
        # (target с пропусками - edge case)
        try:
            result = forge.fit_transform(df)
            assert result is not None
        except Exception as e:
            # Допускаем ошибку валидации
            assert "target" in str(e).lower() or "missing" in str(e).lower()


# ==================== EDGE CASES ====================

class TestEdgeCases:
    """Тесты граничных случаев"""
    
    def test_single_feature(self):
        """Тест с одним признаком"""
        df = pd.DataFrame({
            'feature': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        
        forge = AutoForge(target="target", verbose=False)
        result = forge.fit_transform(df)
        
        assert len(result.data) > 0
    
    def test_many_features(self):
        """Тест с большим числом признаков"""
        np.random.seed(42)
        
        data = {f'feature_{i}': np.random.randn(100) for i in range(50)}
        data['target'] = np.random.choice([0, 1], 100)
        
        df = pd.DataFrame(data)
        
        forge = AutoForge(target="target", verbose=False)
        result = forge.fit_transform(df)
        
        assert len(result.data) > 0
    
    def test_all_missing_in_one_column(self):
        """Тест с полностью пустой колонкой"""
        df = pd.DataFrame({
            'all_missing': [np.nan] * 100,
            'valid': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        
        forge = AutoForge(target="target", verbose=False)
        result = forge.fit_transform(df)
        
        # Полностью пустая колонка должна быть удалена
        assert 'all_missing' not in result.data.columns or result.data['all_missing'].notna().any()
    
    def test_constant_column(self):
        """Тест с константной колонкой"""
        df = pd.DataFrame({
            'constant': [1] * 100,
            'feature': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        
        forge = AutoForge(target="target", verbose=False)
        result = forge.fit_transform(df)
        
        # Константная колонка может быть удалена
        assert len(result.data) > 0
    
    def test_high_cardinality(self):
        """Тест с высокой кардинальностью"""
        df = pd.DataFrame({
            'high_card': [f'category_{i}' for i in range(100)],
            'feature': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        
        forge = AutoForge(
            target="target",
            max_onehot_cardinality=10,
            verbose=False
        )
        result = forge.fit_transform(df)
        
        assert len(result.data) > 0
    
    def test_binary_target_as_string(self):
        """Тест с бинарным target как строкой"""
        df = pd.DataFrame({
            'feature': np.random.randn(100),
            'target': np.random.choice(['yes', 'no'], 100)
        })
        
        forge = AutoForge(target="target", verbose=False)
        result = forge.fit_transform(df)
        
        assert len(result.data) > 0
        assert result.y is not None
    
    def test_small_dataset(self):
        """Тест с маленьким датасетом"""
        df = pd.DataFrame({
            'feature': np.random.randn(20),
            'target': np.random.choice([0, 1], 20)
        })
        
        forge = AutoForge(
            target="target",
            balance=False,  # Отключаем для маленького датасета
            verbose=False
        )
        result = forge.fit_transform(df)
        
        assert len(result.data) == 20


# ==================== MODEL TRAINING TESTS ====================

class TestModelTraining:
    """Тесты интеграции с обучением моделей"""
    
    def test_sklearn_classifier(self, complete_test_df):
        """Тест с sklearn классификатором"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        forge = AutoForge(
            target="target",
            task="classification",
            verbose=False
        )
        
        result = forge.fit_transform(complete_test_df)
        X_train, X_test, y_train, y_test = result.get_splits()
        
        # Обучаем модель
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Должна быть хоть какая-то точность
        assert accuracy > 0.3
        
        print(f"\n✅ Модель обучена, accuracy: {accuracy:.3f}")
    
    def test_sklearn_regressor(self, regression_test_df):
        """Тест с sklearn регрессором"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        
        forge = AutoForge(
            target="target",
            task="regression",
            verbose=False
        )
        
        result = forge.fit_transform(regression_test_df)
        X_train, X_test, y_train, y_test = result.get_splits()
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        # R2 может быть отрицательным для случайных данных
        assert r2 > -1.0
        
        print(f"\n✅ Регрессор обучен, R²: {r2:.3f}")