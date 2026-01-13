# automl_data/adapters/outliers.py
"""
Адаптер обработки выбросов — использует PyOD.
"""

from __future__ import annotations

from typing import Literal
import pandas as pd
import numpy as np

from .base import BaseAdapter
from ..core.container import DataContainer
from ..utils.dependencies import require_package


class OutlierAdapter(BaseAdapter):
    """
    Автоматическое обнаружение и обработка выбросов.
    """
    
    def __init__(
        self,
        method: str = "auto",
        contamination: float = 0.1,
        action: Literal["remove", "clip", "nan", "flag"] = "clip",
        **config
    ):
        super().__init__(name="OutlierHandler", **config)
        self.method = method
        self.contamination = contamination
        self.action = action
        self._fit_info = {}
        
        self._detector = None
        self._numeric_columns: list[str] = []
        self._bounds: dict[str, tuple[float, float]] = {}
    
    def _fit_impl(self, container: DataContainer) -> None:
        require_package("pyod", "pyod")
        
        from pyod.models.ecod import ECOD
        from pyod.models.iforest import IForest
        
        df = container.data
        self._numeric_columns = container.numeric_columns
        
        if not self._numeric_columns:
            return
        
        numeric_data = df[self._numeric_columns].dropna()
        
        if len(numeric_data) < 10:
            return
        
        if self.method == "auto":
            self._detector = ECOD(contamination=self.contamination)
        elif self.method == "iforest":
            self._detector = IForest(
                contamination=self.contamination,
                random_state=42
            )
        else:
            self._detector = ECOD(contamination=self.contamination)
        
        self._detector.fit(numeric_data)
        
        for col in self._numeric_columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            self._bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        self._fit_info.update({
            "detector": self._detector.__class__.__name__, 
            "method": self.method,
            "action": self.action,
            "n_samples": len(container.data),
            "n_features": len(container.numeric_columns)
        })

    
    def _transform_impl(self, container: DataContainer) -> DataContainer:
        if not self._detector or not self._numeric_columns:
            return container
        
        df = container.data.copy()
        numeric_data = df[self._numeric_columns].fillna(
            df[self._numeric_columns].median()
        )
        
        predictions = self._detector.predict(numeric_data)
        outlier_mask = predictions == 1
        outlier_count = outlier_mask.sum()
        
        if outlier_count == 0:
            return container
        
        match self.action:
            case "remove":
                df = df[~outlier_mask]
            case "clip":
                for col in self._numeric_columns:
                    lower, upper = self._bounds[col]
                    df.loc[outlier_mask, col] = df.loc[outlier_mask, col].clip(lower, upper)
            case "nan":
                for col in self._numeric_columns:
                    df.loc[outlier_mask, col] = np.nan
            case "flag":
                df["_is_outlier"] = outlier_mask
        
        container.data = df
        container.recommendations.append({
            "type": "outliers",
            "count": int(outlier_count),
            "action": self.action
        })
        container._sync_internal_state()
        
        return container