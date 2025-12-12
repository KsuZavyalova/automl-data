# automl_data/adapters/balancing.py
"""
Адаптер балансировки классов — использует imbalanced-learn.
"""

from __future__ import annotations

from typing import Any
import pandas as pd
import numpy as np

from .base import BaseAdapter
from ..core.container import DataContainer, ProcessingStage
from ..utils.dependencies import require_package, optional_import


class BalancingAdapter(BaseAdapter):
    """
    Балансировка классов для задач классификации.
    
    Поддерживает:
    - SMOTE (Synthetic Minority Over-sampling Technique)
    - ADASYN (Adaptive Synthetic Sampling)
    - Random Over/Under Sampling
    - SMOTE + Tomek Links
    - SMOTE + ENN
    
    Example:
        >>> balancer = BalancingAdapter(
        ...     strategy="smote",
        ...     target_column="label"
        ... )
        >>> result = balancer.fit_transform(container)
    """
    
    def __init__(
        self,
        strategy: str = "auto",
        target_column: str | None = None,
        imbalance_threshold: float = 0.3,
        sampling_strategy: str | float | dict = "auto",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__(name="Balancing", **kwargs)
        self.strategy = strategy
        self.target_column = target_column
        self.imbalance_threshold = imbalance_threshold
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self._sampler = None
        self._needs_balancing = False
        self._original_distribution: dict = {}
        self._new_distribution: dict = {}
    
    def _fit_impl(self, container: DataContainer) -> None:
        require_package("imblearn", "imbalanced-learn")
        
        from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE
        from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
        from imblearn.combine import SMOTEENN, SMOTETomek
        
        target = container.target_column or self.target_column
        
        if not target or target not in container.data.columns:
            self._fit_info = {"status": "skipped", "reason": "no target column"}
            return
        
        y = container.data[target]
        
        # Проверяем распределение классов
        class_counts = y.value_counts()
        self._original_distribution = class_counts.to_dict()
        
        if len(class_counts) < 2:
            self._fit_info = {"status": "skipped", "reason": "single class"}
            return
        
        # Проверяем степень дисбаланса
        min_ratio = class_counts.min() / class_counts.max()
        
        if min_ratio >= self.imbalance_threshold:
            self._fit_info = {
                "status": "skipped",
                "reason": f"balanced enough (ratio={min_ratio:.2f})"
            }
            return
        
        self._needs_balancing = True
        
        # Проверяем, достаточно ли сэмплов для SMOTE
        min_samples = class_counts.min()
        n_features = len(container.numeric_columns)
        
        # Выбор стратегии
        if self.strategy == "auto":
            if min_samples < 6:
                # Слишком мало для SMOTE
                strategy = "random_over"
            elif n_features > 100:
                # Высокая размерность — ADASYN
                strategy = "adasyn"
            elif min_ratio < 0.1:
                # Сильный дисбаланс — SMOTE + Tomek
                strategy = "smote_tomek"
            else:
                strategy = "smote"
        else:
            strategy = self.strategy
        
        # Создаём sampler
        sampler_map = {
            "smote": lambda: SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                k_neighbors=min(5, min_samples - 1)
            ),
            "borderline_smote": lambda: BorderlineSMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                k_neighbors=min(5, min_samples - 1)
            ),
            "adasyn": lambda: ADASYN(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                n_neighbors=min(5, min_samples - 1)
            ),
            "random_over": lambda: RandomOverSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            ),
            "random_under": lambda: RandomUnderSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            ),
            "smote_tomek": lambda: SMOTETomek(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            "smote_enn": lambda: SMOTEENN(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
        }
        
        try:
            self._sampler = sampler_map.get(strategy, sampler_map["smote"])()
        except Exception as e:
            # Fallback to random oversampling
            self._sampler = RandomOverSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
            strategy = "random_over (fallback)"
        
        self._fit_info = {
            "status": "ready",
            "strategy": strategy,
            "original_ratio": min_ratio,
            "original_distribution": self._original_distribution
        }
    
    def _transform_impl(self, container: DataContainer) -> DataContainer:
        if not self._needs_balancing or self._sampler is None:
            return container
        
        target = container.target_column or self.target_column
        df = container.data.copy()
        
        # Разделяем на X и y
        X = df.drop(columns=[target])
        y = df[target]
        
        # Сохраняем имена колонок
        columns = X.columns.tolist()
        
        # Обрабатываем категориальные колонки для SMOTE
        # SMOTE работает только с числовыми данными
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            # Временно кодируем категории
            X_encoded = X.copy()
            encodings = {}
            
            for col in categorical_cols:
                X_encoded[col], encodings[col] = pd.factorize(X[col])
            
            # Применяем балансировку
            try:
                X_resampled, y_resampled = self._sampler.fit_resample(X_encoded, y)
            except Exception as e:
                container.recommendations.append({
                    "type": "balancing_failed",
                    "error": str(e)
                })
                return container
            
            # Декодируем обратно
            X_result = pd.DataFrame(X_resampled, columns=columns)
            for col in categorical_cols:
                # Округляем индексы и маппим обратно
                indices = X_result[col].round().astype(int).clip(0, len(encodings[col]) - 1)
                X_result[col] = pd.Categorical.from_codes(indices, categories=encodings[col])
        else:
            # Только числовые данные
            try:
                X_resampled, y_resampled = self._sampler.fit_resample(X, y)
                X_result = pd.DataFrame(X_resampled, columns=columns)
            except Exception as e:
                container.recommendations.append({
                    "type": "balancing_failed",
                    "error": str(e)
                })
                return container
        
        # Собираем результат
        df_resampled = X_result.copy()
        df_resampled[target] = y_resampled
        
        # Обновляем распределение
        self._new_distribution = pd.Series(y_resampled).value_counts().to_dict()
        
        container.data = df_resampled.reset_index(drop=True)
        container._sync_internal_state()
        container.stage = ProcessingStage.AUGMENTED
        
        container.recommendations.append({
            "type": "balancing",
            "original_size": len(df),
            "new_size": len(df_resampled),
            "strategy": self._fit_info.get("strategy", "unknown"),
            "original_distribution": self._original_distribution,
            "new_distribution": self._new_distribution
        })
        
        return container