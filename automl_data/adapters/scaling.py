# automl_data/adapters/scaling.py
"""
Основные методы масштабирования признаков.
Использует StandardScaler, RobustScaler, MinMaxScaler.
"""

from __future__ import annotations

from typing import Literal
import pandas as pd
import numpy as np

from .base import BaseAdapter
from ..core.container import DataContainer, ProcessingStage
from ..utils.dependencies import require_package


class ScalingAdapter(BaseAdapter):
    """
    Умный скейлер с автоматическим выбором метода.
    
    Поддерживает:
    - standard: StandardScaler (z-score, для нормальных распределений)
    - robust: RobustScaler (медиана/IQR, при наличии выбросов)
    - minmax: MinMaxScaler [0, 1] (для нейросетей и алгоритмов, требующих bounded values)
    
    Example:
        >>> scaler = ScalingAdapter(strategy="auto")
        >>> result = scaler.fit_transform(container)
    """
    
    def __init__(
        self,
        strategy: Literal["auto", "standard", "robust", "minmax", "none"] = "auto",
        with_mean: bool = True,
        with_std: bool = True,
        quantile_range: tuple[float, float] = (25.0, 75.0),
        **config
    ):
        super().__init__(name="Scaling", **config)
        self.strategy = strategy
        self.with_mean = with_mean
        self.with_std = with_std
        self.quantile_range = quantile_range
        
        self._scaler = None
        self._numeric_cols: list[str] = []
        self._fit_info: dict = {}
        self._actual_strategy = None  
    
    def _fit_impl(self, container: DataContainer) -> None:
        require_package("sklearn", "scikit-learn")
        
        from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
        
        df = container.data
        self._numeric_cols = container.numeric_columns.copy()
        
        target_col = container.target_column
        if target_col and target_col in self._numeric_cols:
            self._numeric_cols.remove(target_col)
        
        if not self._numeric_cols or self.strategy == "none":
            self._actual_strategy = "none"
            return
        
        has_outliers = self._check_outliers(df)
        
        if self.strategy == "auto":
            self._actual_strategy = "robust" if has_outliers else "standard" 
        else:
            self._actual_strategy = self.strategy 
        
        if self._actual_strategy == "standard":
            self._scaler = StandardScaler(
                with_mean=self.with_mean,
                with_std=self.with_std
            )
        
        elif self._actual_strategy == "robust":
            self._scaler = RobustScaler(
                with_centering=self.with_mean,
                with_scaling=self.with_std,
                quantile_range=self.quantile_range
            )
        
        elif self._actual_strategy == "minmax":
            self._scaler = MinMaxScaler()
        
        if self._scaler:
            self._scaler.fit(df[self._numeric_cols])
            
            self._fit_info.update({
                "strategy": self._actual_strategy,
                "has_outliers": has_outliers,
                "scaler_class": self._scaler.__class__.__name__,
                "numeric_cols_count": len(self._numeric_cols),
                "columns_count": len(self._numeric_cols) 
            })
    
    def _check_outliers(self, df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """Проверка на выбросы методом IQR"""
        for col in self._numeric_cols[:15]: 
            if col not in df.columns:
                continue
            
            series = df[col].dropna()
            if len(series) < 10:
                continue
            
            q1, q3 = np.percentile(series, [25, 75])
            iqr = q3 - q1
            
            if iqr == 0:
                continue
            
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            
            outlier_ratio = ((series < lower) | (series > upper)).mean()
            
            if outlier_ratio > threshold:
                return True
        
        return False
    
    def _transform_impl(self, container: DataContainer) -> DataContainer:
        if not self._scaler or not self._numeric_cols:
            return container
        
        target_col = container.target_column
        original_target = None
        if target_col and target_col in container.data.columns:
            original_target = container.data[target_col].copy()
        
        df = container.data.copy()
        
        # Масштабируем числовые колонки
        existing_cols = [c for c in self._numeric_cols if c in df.columns]
        if existing_cols:
            df[existing_cols] = self._scaler.transform(df[existing_cols])
        
        if original_target is not None and target_col in df.columns:
            if not np.array_equal(df[target_col].values, original_target.values):
                df[target_col] = original_target.values
        
        container.data = df
        container._sync_internal_state()
        
        try:
            container.stage = ProcessingStage.SCALED
        except AttributeError:
            if hasattr(ProcessingStage, 'SCALED'):
                container.stage = ProcessingStage.SCALED
            elif hasattr(ProcessingStage, 'TRANSFORMED'):
                container.stage = ProcessingStage.TRANSFORMED
            else:
                container.stage = "SCALED"
        
        container.recommendations.append({
            "type": "scaling",
            "strategy": self._actual_strategy,
            "scaler_class": self._scaler.__class__.__name__,
            "message": f"Applied {self._actual_strategy} scaling"
        })
        
        return container
    
    def inverse_transform(self, container: DataContainer) -> DataContainer:
        """Обратное преобразование (если поддерживается)"""
        if not self._scaler or not hasattr(self._scaler, 'inverse_transform'):
            return container
        
        df = container.data.copy()
        existing_cols = [c for c in self._numeric_cols if c in df.columns]
        
        if existing_cols:
            df[existing_cols] = self._scaler.inverse_transform(df[existing_cols])
        
        container.data = df
        container._sync_internal_state()
        
        return container