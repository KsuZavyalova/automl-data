# automl_data/adapters/feature_cleaner.py
"""
Удаление очевидно бесполезных признаков.
Использует простые, но эффективные эвристики.
"""

from __future__ import annotations

from typing import Literal, Optional
import pandas as pd
import numpy as np
from scipy import stats

from .base import BaseAdapter
from ..core.container import DataContainer, ProcessingStage


class FeatureCleanerAdapter(BaseAdapter):
    """
    Умная очистка признаков с автоматическим удалением проблемных.
    
    Удаляет:
    - Константные признаки (одно значение)
    - Признаки с высоким % пропусков (>threshold)
    - Уникальные идентификаторы (ID колонки)
    - Признаки с нулевой дисперсией
    - Дубликаты признаков (полные копии)
    
    Example:
        >>> cleaner = FeatureCleanerAdapter(
        ...     max_missing_ratio=0.9,
        ...     remove_duplicates=True
        ... )
        >>> result = cleaner.fit_transform(container)
    """
    
    def __init__(
        self,
        max_missing_ratio: float = 0.9,
        remove_duplicates: bool = True,
        correlation_threshold: float = 0.95,
        remove_id_columns: bool = True,
        id_patterns: Optional[list[str]] = None,
        **config
    ):
        super().__init__(name="FeatureCleaning", **config)
        self.max_missing_ratio = max_missing_ratio
        self.remove_duplicates = remove_duplicates
        self.correlation_threshold = correlation_threshold
        self.remove_id_columns = remove_id_columns
        self.id_patterns = id_patterns or ["id", "ID", "Id", "guid", "GUID", "uuid", "UUID"]
        
        self._cols_to_remove: list[str] = []
        self._duplicate_groups: dict = {}
        self._fit_info: dict = {}
    
    def _fit_impl(self, container: DataContainer) -> None:
        df = container.data
        
        # Анализ признаков
        analysis = self._analyze_features(df, container.target_column)
        
        # Определяем признаки для удаления
        self._cols_to_remove = []
        removal_reasons = {}
        
        # 1. Константные и почти константные
        for col in analysis["constant_features"]:
            self._cols_to_remove.append(col)
            removal_reasons[col] = "constant_value"
        
        for col in analysis["low_variance_features"]:
            self._cols_to_remove.append(col)
            removal_reasons[col] = "low_variance"
        
        # 2. Высокий % пропусков
        for col in analysis["high_missing_features"]:
            self._cols_to_remove.append(col)
            removal_reasons[col] = "high_missing"
        
        # 3. ID колонки
        if self.remove_id_columns:
            for col in analysis["id_features"]:
                if col not in self._cols_to_remove:
                    self._cols_to_remove.append(col)
                    removal_reasons[col] = "id_column"
        
        # 4. Дубликаты признаков
        if self.remove_duplicates and analysis["duplicate_groups"]:
            self._duplicate_groups = analysis["duplicate_groups"]
            
            # Оставляем первый признак из каждой группы дубликатов
            for group in analysis["duplicate_groups"].values():
                for col in group[1:]:  # Все кроме первого
                    if col not in self._cols_to_remove:
                        self._cols_to_remove.append(col)
                        removal_reasons[col] = "duplicate_feature"
        
        # 5. Высококоррелированные (опционально)
        if analysis["high_corr_pairs"]:
            # Удаляем один из каждой высококоррелированной пары
            removed_in_corr = set()
            for col1, col2 in analysis["high_corr_pairs"]:
                if col1 not in removed_in_corr and col2 not in removed_in_corr:
                    # Удаляем тот, у которого больше пропусков или меньше уникальных значений
                    if (df[col1].isnull().mean() > df[col2].isnull().mean() or
                        df[col1].nunique() < df[col2].nunique()):
                        col_to_remove = col1
                    else:
                        col_to_remove = col2
                    
                    if col_to_remove not in self._cols_to_remove:
                        self._cols_to_remove.append(col_to_remove)
                        removal_reasons[col_to_remove] = "high_correlation"
                        removed_in_corr.add(col_to_remove)
        
        # Убираем дубликаты
        self._cols_to_remove = list(set(self._cols_to_remove))
        
        self._fit_info.update({
            "analysis": analysis,
            "removal_reasons": removal_reasons,
            "cols_to_remove_count": len(self._cols_to_remove),
            "cols_to_remove": self._cols_to_remove
        })
    
    def _analyze_features(self, df: pd.DataFrame, target_column: Optional[str]) -> dict:
        """Анализ всех признаков"""
        analysis = {
            "constant_features": [],
            "low_variance_features": [],
            "high_missing_features": [],
            "id_features": [],
            "duplicate_groups": {},
            "high_corr_pairs": []
        }
        
        for col in df.columns:
            if col == target_column:
                continue
            
            series = df[col]
            
            # 1. Константные признаки
            if series.nunique() == 1:
                analysis["constant_features"].append(col)
                continue
            
            # 2. Почти константные (более 99% одного значения)
            value_counts = series.value_counts(normalize=True)
            if len(value_counts) > 0 and value_counts.iloc[0] > 0.99:
                analysis["low_variance_features"].append(col)
            
            # 3. Высокий % пропусков
            missing_ratio = series.isnull().mean()
            if missing_ratio > self.max_missing_ratio:
                analysis["high_missing_features"].append(col)
            
            # 4. ID колонки
            if self._is_id_column(col, series):
                analysis["id_features"].append(col)
        
        # 5. Дубликаты признаков
        if self.remove_duplicates:
            analysis["duplicate_groups"] = self._find_duplicate_features(df)
        
        # 6. Высокая корреляция (только для числовых)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1 and target_column not in numeric_cols:
            corr_matrix = df[numeric_cols].corr().abs()
            
            # Находим высококоррелированные пары
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    if col1 != target_column and col2 != target_column:
                        corr = corr_matrix.loc[col1, col2]
                        if corr > self.correlation_threshold:
                            analysis["high_corr_pairs"].append((col1, col2))
        
        return analysis
    
    def _is_id_column(self, col_name: str, series: pd.Series) -> bool:
        """Проверка, является ли колонка ID"""
        # Проверка по названию
        if any(pattern in col_name.lower() for pattern in self.id_patterns):
            return True
        
        # Проверка по данным: все значения уникальны и строка/число
        if (series.nunique() == len(series) and 
            series.dtype in ['object', 'string', 'int64', 'int32']):
            
            # Проверяем формат: guid, email, etc.
            if series.dtype == 'object' or series.dtype == 'string':
                sample = series.dropna().iloc[0] if len(series.dropna()) > 0 else ""
                if isinstance(sample, str):
                    # Проверка на GUID/UUID
                    if len(sample) == 36 and '-' in sample:
                        return True
                    # Проверка на email
                    if '@' in sample and '.' in sample:
                        return True
            
            return True
        
        return False
    
    def _find_duplicate_features(self, df: pd.DataFrame) -> dict:
        """Поиск дубликатов признаков"""
        duplicate_groups = {}
        processed = set()
        
        for i, col1 in enumerate(df.columns):
            if col1 in processed:
                continue
            
            duplicates = [col1]
            
            for j, col2 in enumerate(df.columns[i+1:], i+1):
                if col2 in processed:
                    continue
                
                # Проверяем на равенство значений
                if df[col1].equals(df[col2]):
                    duplicates.append(col2)
                    processed.add(col2)
            
            if len(duplicates) > 1:
                duplicate_groups[col1] = duplicates
                processed.add(col1)
        
        return duplicate_groups
    
    def _transform_impl(self, container: DataContainer) -> DataContainer:
        if not self._cols_to_remove:
            return container
        
        df = container.data.copy()
        
        # Удаляем признаки
        cols_to_keep = [col for col in df.columns if col not in self._cols_to_remove]
        df_cleaned = df[cols_to_keep].copy()
        
        # Обновляем контейнер
        container.data = df_cleaned
        container._sync_internal_state()
        container.stage = ProcessingStage.CLEANED
        
        # Добавляем рекомендации
        container.recommendations.append({
            "type": "feature_cleaning",
            "removed_count": len(self._cols_to_remove),
            "removed_columns": self._cols_to_remove[:10],  # Первые 10
            "reasons": self._fit_info.get("removal_reasons", {}),
            "original_shape": container.shape,
            "cleaned_shape": df_cleaned.shape,
            "message": f"Removed {len(self._cols_to_remove)} problematic features"
        })
        
        return container
    
    def get_removal_report(self) -> dict:
        """Получить отчёт об удалённых признаках"""
        return {
            "removed_columns": self._cols_to_remove,
            "removal_reasons": self._fit_info.get("removal_reasons", {}),
            "duplicate_groups": self._duplicate_groups,
            "analysis": self._fit_info.get("analysis", {})
        }