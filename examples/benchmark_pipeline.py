# examples/benchmark_pipeline.py
"""
Демонстрационный пайплайн сравнения метрик.

Сравнивает качество моделей:
1. На сырых данных (базовая обработка)
2. На данных, обработанных AutoML Data Forge

Поддерживает:
- Табличные данные (классификация, регрессия)
- Текстовые данные (классификация)
- Изображения (классификация)
"""

import sys
from pathlib import Path

# Добавляем корень проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')

import time
from dataclasses import dataclass, field
from typing import Callable
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Модели
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC

# Наша библиотека
from automl_data import AutoForge, ForgeResult


# ==================== Датаклассы для результатов ====================

@dataclass
class MetricsResult:
    """Результаты метрик для одного эксперимента"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    # Для регрессии
    mse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    # Мета
    train_time: float = 0.0
    preprocessing_time: float = 0.0
    
    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


@dataclass
class ExperimentResult:
    """Результат одного эксперимента"""
    dataset_name: str
    task_type: str
    baseline_metrics: MetricsResult
    forge_metrics: MetricsResult
    improvement: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Вычисляем улучшение"""
        if self.task_type == "classification":
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                baseline = getattr(self.baseline_metrics, metric)
                forge = getattr(self.forge_metrics, metric)
                if baseline > 0:
                    self.improvement[metric] = ((forge - baseline) / baseline) * 100
                else:
                    self.improvement[metric] = 0.0
        else:  # regression
            # Для MSE/MAE меньше = лучше
            for metric in ['mse', 'mae']:
                baseline = getattr(self.baseline_metrics, metric)
                forge = getattr(self.forge_metrics, metric)
                if baseline > 0:
                    self.improvement[metric] = ((baseline - forge) / baseline) * 100
            # Для R2 больше = лучше
            baseline_r2 = self.baseline_metrics.r2
            forge_r2 = self.forge_metrics.r2
            if baseline_r2 > 0:
                self.improvement['r2'] = ((forge_r2 - baseline_r2) / abs(baseline_r2)) * 100


# ==================== Генераторы датасетов ====================

class DatasetGenerator(ABC):
    """Базовый класс для генераторов датасетов"""
    
    @abstractmethod
    def generate(self) -> tuple[pd.DataFrame, str, str]:
        """
        Returns:
            (DataFrame, target_column, task_type)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


class TabularClassificationDataset(DatasetGenerator):
    """Табличные данные для классификации"""
    
    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_categories: int = 3,
        missing_rate: float = 0.1,
        imbalance_ratio: float = 0.3,
        noise_level: float = 0.1
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_categories = n_categories
        self.missing_rate = missing_rate
        self.imbalance_ratio = imbalance_ratio
        self.noise_level = noise_level
    
    @property
    def name(self) -> str:
        return f"Tabular_Classification_{self.n_samples}x{self.n_features}"
    
    def generate(self) -> tuple[pd.DataFrame, str, str]:
        np.random.seed(42)
        
        # Генерируем признаки
        data = {}
        
        # Числовые признаки
        n_numeric = self.n_features // 2
        for i in range(n_numeric):
            data[f'numeric_{i}'] = np.random.randn(self.n_samples)
            # Добавляем выбросы
            outlier_idx = np.random.choice(self.n_samples, size=int(self.n_samples * 0.02), replace=False)
            data[f'numeric_{i}'][outlier_idx] *= 10
        
        # Категориальные признаки
        n_categorical = self.n_features - n_numeric
        categories = ['cat_A', 'cat_B', 'cat_C', 'cat_D', 'cat_E']
        for i in range(n_categorical):
            data[f'category_{i}'] = np.random.choice(
                categories[:self.n_categories], 
                self.n_samples
            )
        
        # Целевая переменная (несбалансированная)
        n_positive = int(self.n_samples * self.imbalance_ratio)
        n_negative = self.n_samples - n_positive
        target = np.array([0] * n_negative + [1] * n_positive)
        np.random.shuffle(target)
        data['target'] = target
        
        df = pd.DataFrame(data)
        
        # Добавляем пропуски
        for col in df.columns:
            if col != 'target':
                missing_idx = np.random.choice(
                    self.n_samples, 
                    size=int(self.n_samples * self.missing_rate), 
                    replace=False
                )
                df.loc[missing_idx, col] = np.nan
        
        return df, 'target', 'classification'


class TabularRegressionDataset(DatasetGenerator):
    """Табличные данные для регрессии"""
    
    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 8,
        missing_rate: float = 0.1,
        noise_level: float = 0.2
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.missing_rate = missing_rate
        self.noise_level = noise_level
    
    @property
    def name(self) -> str:
        return f"Tabular_Regression_{self.n_samples}x{self.n_features}"
    
    def generate(self) -> tuple[pd.DataFrame, str, str]:
        np.random.seed(42)
        
        data = {}
        
        # Числовые признаки
        for i in range(self.n_features - 2):
            data[f'feature_{i}'] = np.random.randn(self.n_samples)
        
        # Категориальные
        data['category_1'] = np.random.choice(['low', 'medium', 'high'], self.n_samples)
        data['category_2'] = np.random.choice(['type_A', 'type_B'], self.n_samples)
        
        # Целевая переменная (зависит от признаков + шум)
        target = (
            2 * data['feature_0'] + 
            0.5 * data['feature_1'] - 
            1.5 * data['feature_2'] +
            np.random.randn(self.n_samples) * self.noise_level * 5
        )
        # Добавляем влияние категорий
        target += np.where(np.array(data['category_1']) == 'high', 3, 0)
        data['target'] = target
        
        df = pd.DataFrame(data)
        
        # Добавляем пропуски
        for col in df.columns:
            if col != 'target':
                missing_idx = np.random.choice(
                    self.n_samples, 
                    size=int(self.n_samples * self.missing_rate), 
                    replace=False
                )
                df.loc[missing_idx, col] = np.nan
        
        return df, 'target', 'regression'


class TextClassificationDataset(DatasetGenerator):
    """Текстовые данные для классификации (sentiment analysis)"""
    
    def __init__(self, n_samples: int = 500, imbalance_ratio: float = 0.3):
        self.n_samples = n_samples
        self.imbalance_ratio = imbalance_ratio
    
    @property
    def name(self) -> str:
        return f"Text_Sentiment_{self.n_samples}"
    
    def generate(self) -> tuple[pd.DataFrame, str, str]:
        np.random.seed(42)
        
        positive_templates = [
            "This product is absolutely amazing! I love it!",
            "Great quality and fast shipping. Highly recommend!",
            "Best purchase I've ever made. Five stars!",
            "Exceeded my expectations. Will buy again!",
            "Perfect! Exactly what I was looking for.",
            "Wonderful experience, very satisfied customer.",
            "Outstanding product, works like a charm!",
            "Love love love this! So happy with my purchase.",
            "Fantastic quality for the price. Very impressed!",
            "Couldn't be happier! This is exactly what I needed.",
        ]
        
        negative_templates = [
            "Terrible product. Complete waste of money.",
            "Very disappointed. Does not work as advertised.",
            "Poor quality, broke after one day.",
            "Worst purchase ever. Do not buy this!",
            "Horrible experience. Returning immediately.",
            "Not worth the money. Very cheaply made.",
            "Disappointed with this product. Expected better.",
            "Save your money. This product is garbage.",
            "Awful quality. Falls apart easily.",
            "Regret buying this. Total disappointment.",
        ]
        
        n_positive = int(self.n_samples * (1 - self.imbalance_ratio))
        n_negative = self.n_samples - n_positive
        
        texts = (
            [np.random.choice(positive_templates) for _ in range(n_positive)] +
            [np.random.choice(negative_templates) for _ in range(n_negative)]
        )
        labels = [1] * n_positive + [0] * n_negative
        
        # Добавляем вариативность
        def add_noise(text):
            modifications = [
                lambda t: t.lower(),
                lambda t: t.upper(),
                lambda t: "  " + t + "  ",  # лишние пробелы
                lambda t: t.replace("!", "!!!"),
                lambda t: t,  # без изменений
            ]
            return np.random.choice(modifications)(text)
        
        texts = [add_noise(t) for t in texts]
        
        # Перемешиваем
        indices = np.random.permutation(len(texts))
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
        
        return pd.DataFrame({
            'text': texts,
            'sentiment': labels
        }), 'sentiment', 'text_classification'


class ImbalancedDataset(DatasetGenerator):
    """Сильно несбалансированные данные (95/5)"""
    
    def __init__(self, n_samples: int = 2000):
        self.n_samples = n_samples
    
    @property
    def name(self) -> str:
        return f"Highly_Imbalanced_{self.n_samples}"
    
    def generate(self) -> tuple[pd.DataFrame, str, str]:
        np.random.seed(42)
        
        n_minority = int(self.n_samples * 0.05)
        n_majority = self.n_samples - n_minority
        
        # Мажоритарный класс
        X_majority = np.random.randn(n_majority, 5)
        
        # Миноритарный класс (смещённый)
        X_minority = np.random.randn(n_minority, 5) + 2
        
        X = np.vstack([X_majority, X_minority])
        y = np.array([0] * n_majority + [1] * n_minority)
        
        # Перемешиваем
        indices = np.random.permutation(len(y))
        X = X[indices]
        y = y[indices]
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        
        # Добавляем пропуски
        for col in df.columns[:-1]:
            missing_idx = np.random.choice(len(df), size=int(len(df) * 0.08), replace=False)
            df.loc[missing_idx, col] = np.nan
        
        return df, 'target', 'classification'


# ==================== Базовая обработка (baseline) ====================

def baseline_preprocess(
    df: pd.DataFrame, 
    target: str, 
    task_type: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Минимальная обработка данных (baseline).
    То, что обычно делают вручную.
    """
    df = df.copy()
    y = df.pop(target)
    
    # Определяем типы колонок
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Импьютация числовых
    if numeric_cols:
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Импьютация и кодирование категориальных
    if categorical_cols:
        for col in categorical_cols:
            df[col] = df[col].fillna('missing')
        
        # One-hot encoding
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Масштабирование
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, y


def baseline_text_preprocess(
    df: pd.DataFrame,
    text_col: str,
    target: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Базовая обработка текста (TF-IDF)"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    df = df.copy()
    y = df[target]
    texts = df[text_col].fillna('')
    
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    return pd.DataFrame(X.toarray()), y


# ==================== Обучение и оценка ====================

def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
    model=None
) -> MetricsResult:
    """Обучает модель и вычисляет метрики"""
    
    result = MetricsResult()
    
    # Выбираем модель по умолчанию
    if model is None:
        if task_type in ['classification', 'text_classification']:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Обучаем
    start_time = time.time()
    model.fit(X_train, y_train)
    result.train_time = time.time() - start_time
    
    # Предсказываем
    y_pred = model.predict(X_test)
    
    if task_type in ['classification', 'text_classification']:
        result.accuracy = accuracy_score(y_test, y_pred)
        result.precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        result.recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        result.f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC (только для бинарной классификации)
        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                result.roc_auc = roc_auc_score(y_test, y_proba)
            except Exception:
                result.roc_auc = 0.0
    else:
        result.mse = mean_squared_error(y_test, y_pred)
        result.mae = mean_absolute_error(y_test, y_pred)
        result.r2 = r2_score(y_test, y_pred)
    
    return result


# ==================== Главный класс бенчмарка ====================

class BenchmarkPipeline:
    """
    Пайплайн для сравнения baseline vs AutoForge.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: list[ExperimentResult] = []
    
    def log(self, message: str):
        if self.verbose:
            print(message)
    
    def run_experiment(
        self,
        dataset_generator: DatasetGenerator,
        forge_config: dict = None
    ) -> ExperimentResult:
        """Запускает один эксперимент"""
        
        self.log(f"\n{'='*60}")
        self.log(f"📊 Dataset: {dataset_generator.name}")
        self.log('='*60)
        
        # Генерируем данные
        df, target, task_type = dataset_generator.generate()
        self.log(f"   Shape: {df.shape}, Task: {task_type}")
        self.log(f"   Missing values: {df.isnull().sum().sum()}")
        
        if task_type in ['classification', 'text_classification']:
            class_dist = df[target].value_counts()
            self.log(f"   Class distribution: {dict(class_dist)}")
        
        # ============ BASELINE ============
        self.log(f"\n🔹 Running BASELINE preprocessing...")
        
        start_time = time.time()
        
        if task_type == 'text_classification':
            # Находим текстовую колонку
            text_col = [c for c in df.columns if c != target][0]
            X_baseline, y_baseline = baseline_text_preprocess(df, text_col, target)
        else:
            X_baseline, y_baseline = baseline_preprocess(df, target, task_type)
        
        baseline_preprocess_time = time.time() - start_time
        
        # Train/test split
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
            X_baseline, y_baseline, test_size=0.2, random_state=42, stratify=y_baseline if task_type != 'regression' else None
        )
        
        baseline_metrics = train_and_evaluate(
            X_train_b, X_test_b, y_train_b, y_test_b, task_type
        )
        baseline_metrics.preprocessing_time = baseline_preprocess_time
        
        self.log(f"   Preprocessing time: {baseline_preprocess_time:.2f}s")
        self.log(f"   Training time: {baseline_metrics.train_time:.2f}s")
        
        # ============ AUTOFORGE ============
        self.log(f"\n🔸 Running AUTOFORGE preprocessing...")
        
        start_time = time.time()
        
        # Настройки для AutoForge
        config = {
            'target': target,
            'verbose': False,
        }
        
        if task_type == 'text_classification':
            text_col = [c for c in df.columns if c != target][0]
            config['text_column'] = text_col
        
        if forge_config:
            config.update(forge_config)
        
        try:
            forge = AutoForge(**config)
            result = forge.fit_transform(df)
            
            forge_preprocess_time = time.time() - start_time
            
            # Получаем сплиты
            X_train_f, X_test_f, y_train_f, y_test_f = result.get_splits(test_size=0.2)
            
            forge_metrics = train_and_evaluate(
                X_train_f, X_test_f, y_train_f, y_test_f, task_type
            )
            forge_metrics.preprocessing_time = forge_preprocess_time
            
            self.log(f"   Preprocessing time: {forge_preprocess_time:.2f}s")
            self.log(f"   Training time: {forge_metrics.train_time:.2f}s")
            
        except Exception as e:
            self.log(f"   ⚠️ AutoForge failed: {e}")
            # Используем baseline как fallback
            forge_metrics = baseline_metrics
        
        # ============ СРАВНЕНИЕ ============
        experiment = ExperimentResult(
            dataset_name=dataset_generator.name,
            task_type=task_type,
            baseline_metrics=baseline_metrics,
            forge_metrics=forge_metrics
        )
        
        self._print_comparison(experiment)
        
        self.results.append(experiment)
        return experiment
    
    def _print_comparison(self, exp: ExperimentResult):
        """Выводит сравнение метрик"""
        self.log(f"\n📈 RESULTS COMPARISON:")
        self.log("-" * 50)
        
        if exp.task_type in ['classification', 'text_classification']:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        else:
            metrics = ['mse', 'mae', 'r2']
        
        self.log(f"{'Metric':<15} {'Baseline':>12} {'AutoForge':>12} {'Change':>12}")
        self.log("-" * 50)
        
        for metric in metrics:
            baseline_val = getattr(exp.baseline_metrics, metric)
            forge_val = getattr(exp.forge_metrics, metric)
            change = exp.improvement.get(metric, 0)
            
            # Форматирование
            if metric in ['mse', 'mae']:
                b_str = f"{baseline_val:.4f}"
                f_str = f"{forge_val:.4f}"
            else:
                b_str = f"{baseline_val:.4f}"
                f_str = f"{forge_val:.4f}"
            
            # Цветовой индикатор
            if change > 0:
                change_str = f"+{change:.1f}% ✅"
            elif change < 0:
                change_str = f"{change:.1f}% ❌"
            else:
                change_str = f"{change:.1f}%"
            
            self.log(f"{metric:<15} {b_str:>12} {f_str:>12} {change_str:>12}")
    
    def run_all(self, datasets: list[DatasetGenerator] = None):
        """Запускает все эксперименты"""
        
        if datasets is None:
            datasets = [
                TabularClassificationDataset(n_samples=1000, missing_rate=0.1),
                TabularClassificationDataset(n_samples=500, missing_rate=0.2),
                TabularRegressionDataset(n_samples=1000),
                ImbalancedDataset(n_samples=2000),
                TextClassificationDataset(n_samples=500),
            ]
        
        self.log("\n" + "="*60)
        self.log("🚀 STARTING BENCHMARK PIPELINE")
        self.log("="*60)
        
        for dataset in datasets:
            try:
                self.run_experiment(dataset)
            except Exception as e:
                self.log(f"❌ Error in {dataset.name}: {e}")
        
        self._print_summary()
    
    def _print_summary(self):
        """Итоговая сводка"""
        self.log("\n" + "="*60)
        self.log("📊 FINAL SUMMARY")
        self.log("="*60)
        
        if not self.results:
            self.log("No results to summarize")
            return
        
        # Средние улучшения
        improvements = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'f1': [], 'roc_auc': [], 'r2': []
        }
        
        for exp in self.results:
            for metric, value in exp.improvement.items():
                if metric in improvements:
                    improvements[metric].append(value)
        
        self.log("\n📈 Average improvements across all experiments:")
        for metric, values in improvements.items():
            if values:
                avg = np.mean(values)
                if avg > 0:
                    self.log(f"   {metric}: +{avg:.1f}% ✅")
                else:
                    self.log(f"   {metric}: {avg:.1f}%")
        
        # Таблица результатов
        self.log("\n📋 Detailed results table:")
        self.log("-" * 80)
        self.log(f"{'Dataset':<35} {'Task':<15} {'Baseline F1':<12} {'Forge F1':<12} {'Δ':>8}")
        self.log("-" * 80)
        
        for exp in self.results:
            if exp.task_type in ['classification', 'text_classification']:
                b_metric = exp.baseline_metrics.f1
                f_metric = exp.forge_metrics.f1
                delta = exp.improvement.get('f1', 0)
            else:
                b_metric = exp.baseline_metrics.r2
                f_metric = exp.forge_metrics.r2
                delta = exp.improvement.get('r2', 0)
            
            delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
            self.log(f"{exp.dataset_name:<35} {exp.task_type:<15} {b_metric:<12.4f} {f_metric:<12.4f} {delta_str:>8}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Экспорт результатов в DataFrame"""
        rows = []
        for exp in self.results:
            row = {
                'dataset': exp.dataset_name,
                'task': exp.task_type,
            }
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mse', 'mae', 'r2']:
                row[f'baseline_{metric}'] = getattr(exp.baseline_metrics, metric)
                row[f'forge_{metric}'] = getattr(exp.forge_metrics, metric)
                row[f'improvement_{metric}'] = exp.improvement.get(metric, 0)
            rows.append(row)
        
        return pd.DataFrame(rows)


# ==================== Запуск ====================

def main():
    """Главная функция"""
    
    print("="*60)
    print("   ML DATA FORGE - BENCHMARK PIPELINE")
    print("   Comparing baseline vs automated preprocessing")
    print("="*60)
    
    # Создаём пайплайн
    pipeline = BenchmarkPipeline(verbose=True)
    
    # Определяем датасеты для тестирования
    datasets = [
        # Табличные данные - классификация
        TabularClassificationDataset(
            n_samples=1000,
            n_features=10,
            missing_rate=0.1,
            imbalance_ratio=0.3
        ),
        
        # Табличные данные с большим количеством пропусков
        TabularClassificationDataset(
            n_samples=800,
            n_features=8,
            missing_rate=0.25,
            imbalance_ratio=0.4
        ),
        
        # Регрессия
        TabularRegressionDataset(
            n_samples=1000,
            n_features=8,
            missing_rate=0.15
        ),
        
        # Сильно несбалансированные данные
        ImbalancedDataset(n_samples=2000),
        
        # Текстовые данные
        TextClassificationDataset(
            n_samples=400,
            imbalance_ratio=0.25
        ),
    ]
    
    # Запускаем все эксперименты
    pipeline.run_all(datasets)
    
    # Сохраняем результаты
    results_df = pipeline.to_dataframe()
    results_df.to_csv('benchmark_results.csv', index=False)
    print("\n✅ Results saved to benchmark_results.csv")
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()