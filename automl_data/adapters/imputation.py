# automl_data/adapters/imputation.py
"""
Основные методы импьютации пропусков.
Использует проверенные подходы: Simple, KNN, Iterative (MICE).
"""

from __future__ import annotations

from typing import Literal, Optional
import pandas as pd
import numpy as np

from .base import BaseAdapter
from ..core.container import DataContainer, ProcessingStage
from ..utils.dependencies import require_package
from ..utils.exceptions import ImputationError

class ImputationAdapter(BaseAdapter):
    """
    Умный импьютер с автоматическим выбором метода.
    
    Поддерживает:
    - simple: Mean/Median/Mode (быстро, для маленьких датасетов)
    - knn: K-Nearest Neighbors (для средних датасетов)
    - iterative: MICE (Multiple Imputation by Chained Equations, для сложных случаев)
    
    Example:
        >>> imputer = ImputationAdapter(strategy="auto")
        >>> result = imputer.fit_transform(container)
    """
    
    def __init__(
        self,
        strategy: Literal["auto", "simple", "knn", "iterative", "none"] = "auto",
        numeric_strategy: Literal["mean", "median", "most_frequent"] = "median",
        categorical_strategy: str = "most_frequent",
        constant_value: Optional[float] = None,
        n_neighbors: int = 5,
        max_iter: int = 10,
        random_state: int = 42,
        **config
    ):
        super().__init__(name="Imputation", **config)
        self.strategy = strategy
        self._actual_strategy = None
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.constant_value = constant_value
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.random_state = random_state
        
        self._imputer = None
        self._numeric_imputer = None
        self._categorical_fill_values: dict[str, str] = {}
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._fit_info: dict = {}
    
    def _fit_impl(self, container: DataContainer) -> None:
        df = container.data
        
        self._numeric_cols = container.numeric_columns.copy()
        self._categorical_cols = container.categorical_columns.copy()
        
        target_col = container.target_column
        if target_col:
            if target_col in self._numeric_cols:
                self._numeric_cols.remove(target_col)
            if target_col in self._categorical_cols:
                self._categorical_cols.remove(target_col)
        
        missing_info = self._analyze_missingness(df)
        
        if self.strategy == "auto":
            self._actual_strategy = self._select_best_strategy(missing_info, df)
        else:
            self._actual_strategy = self.strategy
            
        self._fit_info.update({
            "strategy": self.strategy,
            "missing_info": missing_info,
            "numeric_cols_count": len(self._numeric_cols),
            "categorical_cols_count": len(self._categorical_cols)
        })
        
        if self.strategy != "none":
            self._fit_imputer(df)
    
    def _select_best_strategy(self, missing_info: dict, df: pd.DataFrame) -> str:
        """Автовыбор лучшего метода импьютации"""
        missing_ratio = missing_info["total_missing_ratio"]
        n_samples = len(df)
        
        if missing_ratio < 0.05:  # Мало пропусков
            return "simple"
        elif missing_ratio < 0.3:  # Умеренное количество пропусков
            if n_samples > 1000 and len(self._numeric_cols) > 5:
                return "iterative"  # MICE для больших датасетов
            else:
                return "knn"  # KNN для маленьких/средних
        else:  # Много пропусков (>30%)
            return "simple"  # Простой метод, чтобы не усложнять
    
    def _fit_imputer(self, df: pd.DataFrame):
        """Обучение импьютера"""
        require_package("sklearn", "scikit-learn")
        
        from sklearn.impute import SimpleImputer, KNNImputer
        
        if self.strategy == "simple":
            # Простая импьютация
            if self._numeric_cols:
                self._numeric_imputer = SimpleImputer(
                    strategy=self.numeric_strategy,
                    fill_value=self.constant_value
                )
                self._numeric_imputer.fit(df[self._numeric_cols])
            
            # Категориальная импьютация (mode)
            for col in self._categorical_cols:
                if df[col].isnull().any():
                    mode = df[col].mode()
                    self._categorical_fill_values[col] = (
                        mode.iloc[0] if len(mode) > 0 else "MISSING"
                    )
        
        elif self.strategy == "knn":
            # KNN импьютация
            if self._numeric_cols:
                self._numeric_imputer = KNNImputer(
                    n_neighbors=min(self.n_neighbors, max(1, len(df) - 1)),
                    weights="distance"
                )
                self._numeric_imputer.fit(df[self._numeric_cols])
            
            # Для категориальных используем simple (mode)
            for col in self._categorical_cols:
                if df[col].isnull().any():
                    mode = df[col].mode()
                    self._categorical_fill_values[col] = (
                        mode.iloc[0] if len(mode) > 0 else "MISSING"
                    )
        
        elif self.strategy == "iterative":
            # MICE импьютация
            try:
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                from sklearn.linear_model import BayesianRidge
                
                self._numeric_imputer = IterativeImputer(
                    estimator=BayesianRidge(),
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    initial_strategy=self.numeric_strategy
                )
                
                if self._numeric_cols:
                    self._numeric_imputer.fit(df[self._numeric_cols])
                
                # Для категориальных используем simple
                for col in self._categorical_cols:
                    if df[col].isnull().any():
                        mode = df[col].mode()
                        self._categorical_fill_values[col] = (
                            mode.iloc[0] if len(mode) > 0 else "MISSING"
                        )
                        
            except ImportError:
                # Fallback to KNN
                self.strategy = "knn"
                self._fit_imputer(df)
    
    def _transform_impl(self, container: DataContainer) -> DataContainer:
        if self.strategy == "none":
            return container
        
        target_col = container.target_column
        original_target = None
        if target_col and target_col in container.data.columns:
            original_target = container.data[target_col].copy()
        
        df = container.data.copy()
        
        # Импьютация числовых
        if self._numeric_imputer and self._numeric_cols:
            existing_cols = [c for c in self._numeric_cols if c in df.columns]
            if existing_cols:
                df[existing_cols] = self._numeric_imputer.transform(df[existing_cols])
        
        # Импьютация категориальных
        for col, fill_value in self._categorical_fill_values.items():
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(fill_value)
        
        if original_target is not None and target_col in df.columns:
            if not np.array_equal(df[target_col].values, original_target.values):
                df[target_col] = original_target.values
        
        container.data = df
        container._sync_internal_state()
        container.stage = ProcessingStage.IMPUTED
        
        container.recommendations.append({
            "type": "imputation",
            "strategy": self.strategy,
            "missing_before": self._fit_info["missing_info"]["total_missing"],
            "missing_after": df.isnull().sum().sum(),
            "message": f"Applied {self.strategy} imputation"
        })
        
        return container
    
    def _analyze_missingness(self, df: pd.DataFrame) -> dict:
        """Анализ пропусков"""
        missing_matrix = df.isnull()
        total_missing = missing_matrix.sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        missing_ratio = total_missing / total_cells
        
        missing_by_col = missing_matrix.sum()
        cols_with_missing = missing_by_col[missing_by_col > 0].index.tolist()
    
        missing_type = "MCAR" if missing_ratio < 0.05 else "MAR/MNAR"
        
        return {
            "total_missing": int(total_missing),
            "total_missing_ratio": missing_ratio,
            "cols_with_missing": cols_with_missing,
            "missing_type": missing_type
        }