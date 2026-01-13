# automl_data/adapters/encoding.py
"""
Адаптер кодирования категориальных переменных — использует category_encoders.
"""

from __future__ import annotations

from typing import Any, Literal
import pandas as pd
import numpy as np

from .base import BaseAdapter
from ..core.container import DataContainer, ProcessingStage
from ..utils.dependencies import require_package, optional_import


class EncodingAdapter(BaseAdapter):
    """
    Умное кодирование категориальных переменных.
    
    Автоматически выбирает метод для каждой колонки:
    - 2 категории → Binary
    - <= N категорий → OneHot
    - > N категорий + есть target → Target Encoding
    - > N категорий без target → Ordinal
    
    Example:
        >>> encoder = EncodingAdapter(
        ...     strategy="auto",
        ...     max_onehot_cardinality=10
        ... )
        >>> result = encoder.fit_transform(container)
    """
    
    def __init__(
        self,
        strategy: Literal["auto", "onehot", "target", "ordinal", "binary", "woe"] = "auto",
        target_column: str | None = None,
        max_onehot_cardinality: int = 10,
        handle_unknown: str = "value",
        handle_missing: str = "value",
        min_samples_leaf: int = 1,
        smoothing: float = 1.0,
        **config
    ):
        super().__init__(name="Encoding", **config)
        self.strategy = strategy
        self.target_column = target_column
        self.max_cardinality = max_onehot_cardinality
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        
        self._encoders: dict[str, Any] = {}
        self._strategies: dict[str, str] = {}
        self._columns_to_encode: list[str] = []
    
    def _fit_impl(self, container: DataContainer) -> None:
        require_package("category_encoders", "category-encoders")
        
        import category_encoders as ce
        
        df = container.data
        target = container.target_column or self.target_column
        
        self._columns_to_encode = container.categorical_columns.copy()
        
        if target and target in self._columns_to_encode:
            self._columns_to_encode.remove(target)
        
        if not self._columns_to_encode:
            self._fit_info["status"] = "no categorical columns"
            return
        
        for col in self._columns_to_encode:
            n_unique = df[col].nunique()
            
            if self.strategy == "auto":
                if n_unique == 2:
                    strat = "binary"
                elif n_unique <= self.max_cardinality:
                    strat = "onehot"
                elif target and target in df.columns and self._is_classification_target(df[target]):
                    strat = "target"
                else:
                    strat = "ordinal"
            else:
                strat = self.strategy
            
            self._strategies[col] = strat
            
            try:
                if strat == "onehot":
                    encoder = ce.OneHotEncoder(
                        cols=[col],
                        handle_unknown=self.handle_unknown,
                        handle_missing=self.handle_missing,
                        use_cat_names=True
                    )
                elif strat == "binary":
                    encoder = ce.BinaryEncoder(
                        cols=[col],
                        handle_unknown=self.handle_unknown,
                        handle_missing=self.handle_missing
                    )
                elif strat == "target":
                    encoder = ce.TargetEncoder(
                        cols=[col],
                        handle_unknown=self.handle_unknown,
                        handle_missing=self.handle_missing,
                        min_samples_leaf=self.min_samples_leaf,
                        smoothing=self.smoothing
                    )
                elif strat == "woe":
                    encoder = ce.WOEEncoder(
                        cols=[col],
                        handle_unknown=self.handle_unknown,
                        handle_missing=self.handle_missing
                    )
                else:  # ordinal
                    encoder = ce.OrdinalEncoder(
                        cols=[col],
                        handle_unknown=self.handle_unknown,
                        handle_missing=self.handle_missing
                    )
                
                # Fit
                if strat in ["target", "woe"] and target and target in df.columns:
                    encoder.fit(df[[col]], df[target])
                else:
                    encoder.fit(df[[col]])
                
                self._encoders[col] = encoder
                
            except Exception as e:
                encoder = ce.OrdinalEncoder(
                    cols=[col],
                    handle_unknown="value",
                    handle_missing="value"
                )
                encoder.fit(df[[col]])
                self._encoders[col] = encoder
                self._strategies[col] = "ordinal (fallback)"
        
        self._fit_info["strategies"] = self._strategies.copy()
        self._fit_info["encoded_columns"] = len(self._encoders)
    
    def _transform_impl(self, container: DataContainer) -> DataContainer:
        if not self._encoders:
            return container
    
        target = container.target_column or self.target_column
        target_series = None
        if target and target in container.data.columns:
            target_series = container.data[target].copy()
        
        df = container.data.copy()

        target = container.target_column or self.target_column
        
        for col, encoder in self._encoders.items():
            if col not in df.columns:
                continue
            
            strat = self._strategies.get(col, "ordinal")
            
            try:
                if strat in ["target", "woe"] and target and target in df.columns:
                    encoded = encoder.transform(df[[col]])
                else:
                    encoded = encoder.transform(df[[col]])
                
                df = df.drop(columns=[col])
                
                for new_col in encoded.columns:
                    if new_col not in df.columns:
                        df[new_col] = encoded[new_col].values
            
            except Exception as e:
                container.recommendations.append({
                    "type": "encoding_warning",
                    "column": col,
                    "error": str(e)
                })
        
        if target_series is not None:
            if target not in df.columns:
                df[target] = target_series.values
            else:
                current_target = df[target]
                if hasattr(current_target, 'values'):
                    if not np.array_equal(current_target.values, target_series.values):
                        df[target] = target_series.values
        
        container.data = df
        
        container._sync_internal_state()
        
        container.stage = ProcessingStage.TRANSFORMED
        
        return container
    
    def _is_classification_target(self, series: pd.Series) -> bool:
        """Проверка, является ли target классификационным"""
        if series.dtype == 'object' or series.dtype.name == 'category':
            return True
        
        n_unique = series.nunique()
        unique_ratio = n_unique / len(series)
        
        # Если мало уникальных значений — классификация
        return n_unique <= 50 and unique_ratio < 0.1
    
    def get_feature_names(self) -> list[str]:
        """Получить имена закодированных признаков"""
        names = []
        for col, encoder in self._encoders.items():
            if hasattr(encoder, 'get_feature_names'):
                names.extend(encoder.get_feature_names())
            elif hasattr(encoder, 'feature_names_out_'):
                names.extend(encoder.feature_names_out_)
            else:
                names.append(col)
        return names