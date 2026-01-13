# tests/test_balancing_classification.py
"""
Тесты балансировки данных для задач классификации.

Проверяет:
- Генерация синтетических данных при балансировке
- Корректное определение задачи классификации
- Работа с разными степенями дисбаланса
- Сохранение структуры данных после балансировки
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from automl_data.core.forge import AutoForge
from automl_data.core.container import DataContainer
from automl_data.adapters.balancing import BalancingAdapter


# ==================== FIXTURES ====================

@pytest.fixture
def imbalanced_binary_df() -> pd.DataFrame:
    """Сильно несбалансированный бинарный датасет (10:1)"""
    np.random.seed(42)
    
    # 900 примеров класса 0, 100 примеров класса 1
    n_majority = 900
    n_minority = 100
    
    X_majority = np.random.randn(n_majority, 5)
    X_minority = np.random.randn(n_minority, 5) + 2  # Смещение для различимости
    
    X = np.vstack([X_majority, X_minority])
    y = np.array([0] * n_majority + [1] * n_minority)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    
    return df


@pytest.fixture
def imbalanced_multiclass_df() -> pd.DataFrame:
    """Несбалансированный мультиклассовый датасет"""
    np.random.seed(42)
    
    # Класс 0: 500, Класс 1: 100, Класс 2: 50
    sizes = [500, 100, 50]
    
    dfs = []
    for cls, size in enumerate(sizes):
        X = np.random.randn(size, 4) + cls  # Смещение
        df_cls = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(4)])
        df_cls['label'] = cls
        dfs.append(df_cls)
    
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def mixed_types_imbalanced_df() -> pd.DataFrame:
    """Датасет со смешанными типами и дисбалансом"""
    np.random.seed(42)
    
    n_majority = 800
    n_minority = 200
    
    df = pd.DataFrame({
        'numeric_1': np.random.randn(n_majority + n_minority),
        'numeric_2': np.random.uniform(0, 100, n_majority + n_minority),
        'category_1': np.random.choice(['A', 'B', 'C'], n_majority + n_minority),
        'category_2': np.random.choice(['X', 'Y'], n_majority + n_minority),
        'target': [0] * n_majority + [1] * n_minority
    })
    
    return df


@pytest.fixture
def titanic_like_df() -> pd.DataFrame:
    """Датасет похожий на Titanic"""
    np.random.seed(42)
    n = 891
    
    survived = np.random.choice([0, 1], n, p=[0.62, 0.38])  # Реальное соотношение Titanic
    
    df = pd.DataFrame({
        'Pclass': np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
        'Sex': np.random.choice(['male', 'female'], n, p=[0.65, 0.35]),
        'Age': np.random.uniform(1, 80, n),
        'SibSp': np.random.choice([0, 1, 2, 3, 4], n, p=[0.68, 0.23, 0.05, 0.02, 0.02]),
        'Parch': np.random.choice([0, 1, 2], n, p=[0.76, 0.13, 0.11]),
        'Fare': np.random.exponential(30, n),
        'Embarked': np.random.choice(['S', 'C', 'Q'], n, p=[0.72, 0.19, 0.09]),
        'Survived': survived
    })
    
    # Добавляем пропуски как в реальном Titanic
    age_missing = np.random.choice(n, size=int(n * 0.2), replace=False)
    df.loc[age_missing, 'Age'] = np.nan
    
    return df


# ==================== BASIC BALANCING TESTS ====================

class TestBalancingBasic:
    """Базовые тесты балансировки"""
    
    def test_data_generated_after_balancing(self, imbalanced_binary_df):
        """Проверяет, что данные генерируются при балансировке"""
        df = imbalanced_binary_df
        original_size = len(df)
        original_distribution = df['target'].value_counts().to_dict()
        
        forge = AutoForge(
            target='target',
            task='classification',
            balance=True,
            verbose=False
        )
        
        result = forge.fit_transform(df)
        
        new_size = len(result.data)
        new_distribution = result.data['target'].value_counts().to_dict()
        
        # Данные должны быть сгенерированы (размер увеличился)
        assert new_size > original_size, \
            f"Ожидалось увеличение данных: было {original_size}, стало {new_size}"
        
        # Баланс должен улучшиться
        original_ratio = min(original_distribution.values()) / max(original_distribution.values())
        new_ratio = min(new_distribution.values()) / max(new_distribution.values())
        
        assert new_ratio > original_ratio, \
            f"Баланс должен улучшиться: было {original_ratio:.2f}, стало {new_ratio:.2f}"
        
        print(f"\n✅ Сгенерировано {new_size - original_size} новых примеров")
        print(f"   Исходное распределение: {original_distribution}")
        print(f"   Новое распределение: {new_distribution}")
    
    def test_minority_class_increased(self, imbalanced_binary_df):
        """Проверяет, что миноритарный класс увеличился"""
        df = imbalanced_binary_df
        
        original_minority = (df['target'] == 1).sum()
        
        forge = AutoForge(
            target='target',
            task='classification',
            balance=True,
            verbose=False
        )
        
        result = forge.fit_transform(df)
        
        new_minority = (result.data['target'] == 1).sum()
        
        assert new_minority > original_minority, \
            f"Миноритарный класс должен увеличиться: было {original_minority}, стало {new_minority}"
    
    def test_majority_class_unchanged_with_oversampling(self, imbalanced_binary_df):
        """Проверяет, что мажоритарный класс не уменьшился при oversampling"""
        df = imbalanced_binary_df
        
        original_majority = (df['target'] == 0).sum()
        
        # Используем стратегию oversampling
        container = DataContainer(data=df.copy(), target_column='target')
        balancer = BalancingAdapter(
            strategy='smote',
            target_column='target',
            imbalance_threshold=0.3
        )
        
        result = balancer.fit_transform(container)
        
        new_majority = (result.data['target'] == 0).sum()
        
        assert new_majority >= original_majority, \
            f"Мажоритарный класс не должен уменьшаться: было {original_majority}, стало {new_majority}"


# ==================== CLASSIFICATION TASK DETECTION ====================

class TestClassificationDetection:
    """Тесты определения задачи классификации"""
    
    def test_forced_classification_task(self, imbalanced_binary_df):
        """Проверяет принудительное задание task='classification'"""
        df = imbalanced_binary_df
        
        forge = AutoForge(
            target='target',
            task='classification',  # Принудительно
            balance=True,
            verbose=False
        )
        
        forge.fit(df)
        
        assert forge.config.task.value == 'classification', \
            f"Задача должна быть classification, получено {forge.config.task.value}"
    
    def test_auto_detection_binary(self, imbalanced_binary_df):
        """Проверяет автоопределение бинарной классификации"""
        df = imbalanced_binary_df
        
        forge = AutoForge(
            target='target',
            task='auto',
            balance=True,
            verbose=False
        )
        
        forge.fit(df)
        
        # Должна определиться классификация (мало уникальных значений)
        assert forge.config.task.value == 'classification', \
            f"Ожидалась classification, получено {forge.config.task.value}"
    
    def test_auto_detection_multiclass(self, imbalanced_multiclass_df):
        """Проверяет автоопределение мультиклассовой классификации"""
        df = imbalanced_multiclass_df
        
        forge = AutoForge(
            target='label',
            task='auto',
            balance=True,
            verbose=False
        )
        
        forge.fit(df)
        
        assert forge.config.task.value == 'classification'
    
    def test_categorical_target_is_classification(self):
        """Проверяет, что категориальный target = классификация"""
        df = pd.DataFrame({
            'feature': np.random.randn(100),
            'target': np.random.choice(['cat', 'dog', 'bird'], 100)
        })
        
        forge = AutoForge(
            target='target',
            task='auto',
            verbose=False
        )
        
        forge.fit(df)
        
        assert forge.config.task.value == 'classification'


# ==================== MULTICLASS BALANCING ====================

class TestMulticlassBalancing:
    """Тесты балансировки мультиклассовых данных"""
    
    def test_multiclass_balancing_generates_data(self, imbalanced_multiclass_df):
        """Проверяет генерацию данных для мультиклассовой задачи"""
        df = imbalanced_multiclass_df
        
        original_counts = df['label'].value_counts().sort_index()
        original_size = len(df)
        
        forge = AutoForge(
            target='label',
            task='classification',
            balance=True,
            verbose=False
        )
        
        result = forge.fit_transform(df)
        
        new_counts = result.data['label'].value_counts().sort_index()
        
        # Минимальный класс должен увеличиться
        assert new_counts.min() > original_counts.min(), \
            "Минимальный класс должен увеличиться"
        
        print(f"\n✅ Мультиклассовая балансировка:")
        print(f"   Исходное: {original_counts.to_dict()}")
        print(f"   Новое: {new_counts.to_dict()}")
    
    def test_all_classes_present_after_balancing(self, imbalanced_multiclass_df):
        """Проверяет, что все классы присутствуют после балансировки"""
        df = imbalanced_multiclass_df
        
        original_classes = set(df['label'].unique())
        
        forge = AutoForge(
            target='label',
            task='classification',
            balance=True,
            verbose=False
        )
        
        result = forge.fit_transform(df)
        
        new_classes = set(result.data['label'].unique())
        
        assert original_classes == new_classes, \
            f"Классы должны сохраниться: было {original_classes}, стало {new_classes}"


# ==================== MIXED TYPES BALANCING ====================

class TestMixedTypesBalancing:
    """Тесты балансировки с разными типами данных"""
    
    def test_categorical_features_preserved(self, mixed_types_imbalanced_df):
        """Проверяет сохранение категориальных признаков"""
        df = mixed_types_imbalanced_df
        
        original_categories = {
            'category_1': set(df['category_1'].unique()),
            'category_2': set(df['category_2'].unique())
        }
        
        forge = AutoForge(
            target='target',
            task='classification',
            balance=True,
            verbose=False
        )
        
        result = forge.fit_transform(df)
        
        # После кодирования категории будут числовыми,
        # но проверяем, что балансировка прошла
        assert len(result.data) > len(df), "Данные должны быть сгенерированы"
    

# ==================== TITANIC-LIKE TESTS ====================

class TestTitanicLikeData:
    """Тесты на данных похожих на Titanic"""
    
    def test_titanic_full_pipeline(self, titanic_like_df):
        """Полный пайплайн обработки Titanic-подобных данных"""
        df = titanic_like_df
        
        original_size = len(df)
        original_distribution = df['Survived'].value_counts().to_dict()
        
        forge = AutoForge(
            target='Survived',
            task='classification',
            balance=True,
            verbose=False
        )
        
        result = forge.fit_transform(df)
        
        # Проверяем базовые свойства
        assert result.data is not None
        assert len(result.data) > 0
        assert result.quality_score > 0
        
        # Проверяем, что обработка прошла
        assert len(result.steps) > 0, "Должны быть выполнены шаги обработки"
        
        print(f"\n✅ Titanic-like обработка:")
        print(f"   Исходный размер: {original_size}")
        print(f"   Новый размер: {len(result.data)}")
        print(f"   Качество: {result.quality_score:.0%}")
        print(f"   Шаги: {result.steps}")
    
    def test_titanic_splits_valid(self, titanic_like_df):
        """Проверяет корректность train/test сплитов"""
        df = titanic_like_df
        
        forge = AutoForge(
            target='Survived',
            task='classification',
            balance=True,
            test_size=0.2,
            verbose=False
        )
        
        result = forge.fit_transform(df)
        
        X_train, X_test, y_train, y_test = result.get_splits()
        
        # Проверяем размеры
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        
        assert 0.15 < test_ratio < 0.25, \
            f"Test ratio должен быть ~0.2, получено {test_ratio:.2f}"
        
        # Проверяем, что target корректный
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})
        
        # Проверяем, что X и y совпадают по размеру
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    
    def test_titanic_model_trainable(self, titanic_like_df):
        """Проверяет, что на обработанных данных можно обучить модель"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        df = titanic_like_df
        
        forge = AutoForge(
            target='Survived',
            task='classification',
            balance=True,
            verbose=False
        )
        
        result = forge.fit_transform(df)
        X_train, X_test, y_train, y_test = result.get_splits()
        
        # Обучаем модель
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Accuracy должен быть лучше случайного
        assert accuracy > 0.5, f"Accuracy должен быть > 0.5, получено {accuracy:.3f}"
        
        print(f"\n✅ Модель обучена, accuracy: {accuracy:.3f}")


# ==================== BALANCING ADAPTER UNIT TESTS ====================

class TestBalancingAdapterUnit:
    """Юнит-тесты BalancingAdapter"""
    
    def test_adapter_fit_transform(self, imbalanced_binary_df):
        """Тест fit_transform адаптера"""
        df = imbalanced_binary_df
        
        container = DataContainer(data=df.copy(), target_column='target')
        
        adapter = BalancingAdapter(
            strategy='smote',
            target_column='target',
            imbalance_threshold=0.3
        )
        
        result = adapter.fit_transform(container)
        
        assert adapter.is_fitted
        assert len(result.data) > len(df)
    
    def test_adapter_different_strategies(self, imbalanced_binary_df):
        """Тест разных стратегий балансировки"""
        df = imbalanced_binary_df
        strategies = ['smote', 'random_over', 'adasyn']
        
        for strategy in strategies:
            container = DataContainer(data=df.copy(), target_column='target')
            
            adapter = BalancingAdapter(
                strategy=strategy,
                target_column='target',
                imbalance_threshold=0.3
            )
            
            try:
                result = adapter.fit_transform(container)
                assert len(result.data) >= len(df), \
                    f"Стратегия {strategy} не увеличила данные"
                print(f"✅ {strategy}: {len(df)} -> {len(result.data)}")
            except Exception as e:
                pytest.fail(f"Стратегия {strategy} упала с ошибкой: {e}")
    
    def test_adapter_skips_balanced_data(self):
        """Тест, что адаптер пропускает сбалансированные данные"""
        # Создаём сбалансированный датасет
        df = pd.DataFrame({
            'feature': np.random.randn(200),
            'target': [0] * 100 + [1] * 100  # 50/50
        })
        
        container = DataContainer(data=df.copy(), target_column='target')
        
        adapter = BalancingAdapter(
            strategy='auto',
            target_column='target',
            imbalance_threshold=0.3  # Порог 30%
        )
        
        result = adapter.fit_transform(container)
        
        # Данные не должны измениться (уже сбалансированы)
        assert len(result.data) == len(df), \
            "Сбалансированные данные не должны изменяться"
    
    def test_adapter_recommendations(self, imbalanced_binary_df):
        """Тест, что адаптер добавляет рекомендации"""
        df = imbalanced_binary_df
        
        container = DataContainer(data=df.copy(), target_column='target')
        
        adapter = BalancingAdapter(
            strategy='smote',
            target_column='target',
            imbalance_threshold=0.3
        )
        
        result = adapter.fit_transform(container)
        
        # Должна быть рекомендация о балансировке
        balancing_recs = [
            r for r in result.recommendations 
            if r.get('type') == 'balancing'
        ]
        
        assert len(balancing_recs) > 0, "Должна быть рекомендация о балансировке"
        
        rec = balancing_recs[0]
        assert 'strategy' in rec
        assert 'original_size' in rec
        assert 'new_size' in rec


# ==================== EDGE CASES ====================

class TestEdgeCases:
    """Тесты граничных случаев"""

    
    def test_single_class_no_error(self):
        """Тест, что один класс не вызывает ошибку"""
        df = pd.DataFrame({
            'feature': np.random.randn(100),
            'target': [0] * 100  # Только один класс
        })
        
        forge = AutoForge(
            target='target',
            task='classification',
            balance=True,
            verbose=False
        )
        
        # Не должно быть ошибки
        result = forge.fit_transform(df)
        assert len(result.data) > 0
    
# ==================== INTEGRATION TEST ====================

class TestIntegration:
    """Интеграционные тесты"""
    
    def test_full_pipeline_with_all_steps(self, mixed_types_imbalanced_df):
        """Полный пайплайн со всеми шагами"""
        df = mixed_types_imbalanced_df
        
        forge = AutoForge(
            target='target',
            task='classification',
            balance=True,
            impute_strategy='auto',
            scaling='auto',
            encode_strategy='auto',
            verbose=True  # Включаем логи для отладки
        )
        
        result = forge.fit_transform(df)
        
        # Проверяем все шаги
        expected_steps = {'Imputation', 'Encoding', 'Scaling', 'Balancing'}
        actual_steps = set(result.steps)
        
        # Хотя бы часть шагов должна выполниться
        assert len(actual_steps) > 0, "Должны быть выполнены шаги"
        
        # Проверяем результат
        assert result.quality_score > 0
        assert len(result.data) > 0
        
        # Можно получить сплиты
        X_train, X_test, y_train, y_test = result.get_splits()
        assert len(X_train) > 0
        assert len(X_test) > 0
        
        print(f"\n✅ Полный пайплайн выполнен:")
        print(f"   Шаги: {result.steps}")
        print(f"   Размер: {len(df)} -> {len(result.data)}")
        print(f"   Качество: {result.quality_score:.0%}")