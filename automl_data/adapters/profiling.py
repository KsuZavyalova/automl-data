# automl_data/adapters/profiling.py
"""
Адаптер профилирования — использует ydata-profiling.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

from .base import TransformOnlyAdapter
from ..core.container import DataContainer, ProcessingStage
from ..utils.dependencies import require_package, optional_import


class ProfilerAdapter(TransformOnlyAdapter):
    """
    Умное профилирование данных с автоматическим выбором метода.
    
    Генерирует детальный отчёт:
    - Статистики по колонкам (основные метрики)
    - Корреляции (матрица корреляций)
    - Пропуски (паттерны и распределение)
    - Дубликаты (полные и частичные)
    - Распределения (гистограммы, QQ-plot)
    - Алерты о проблемах с приоритетами
    
    Example:
        >>> profiler = ProfilerAdapter(mode="minimal")
        >>> container = profiler.transform(container)
        >>> profiler.save_report("report.html")
    """
    
    def __init__(
        self,
        mode: Literal["minimal", "full", "explorative"] = "minimal",
        correlation_threshold: float = 0.8,
        missing_threshold: float = 0.5,
        include_correlations: bool = True,
        include_interactions: bool = False,
        sensitive_features: Optional[list[str]] = None,
        title: str = "Data Profile Report",
        **kwargs
    ):
        super().__init__(name="Profiler", **kwargs)
        self.mode = mode
        self.correlation_threshold = correlation_threshold
        self.missing_threshold = missing_threshold
        self.include_correlations = include_correlations
        self.include_interactions = include_interactions
        self.sensitive_features = sensitive_features or []
        self.title = title
        
        self._profile = None
        self._profile_dict: dict = {}
        self._alerts: list[dict] = []
    
    def _transform_impl(self, container: DataContainer) -> DataContainer:
        # Сохраняем timestamp начала профилирования
        start_time = datetime.now()
        
        # Пробуем использовать ydata-profiling если доступен
        ydata = optional_import("ydata_profiling")
        
        if ydata and self.mode != "minimal":
            container = self._profile_with_ydata(container)
        else:
            # Используем наш быстрый профилировщик
            container = self._profile_fast(container)
        
        # Добавляем тайминг
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Обновляем профиль
        if container.profile:
            container.profile["profiling_duration"] = duration
            container.profile["profiling_mode"] = self.mode
            container.profile["profiling_timestamp"] = start_time.isoformat()
        
        container.stage = ProcessingStage.PROFILED
        
        # Добавляем рекомендации на основе профиля
        self._add_recommendations(container)
        
        return container
    
    def _profile_with_ydata(self, container: DataContainer) -> DataContainer:
        """Профилирование через ydata-profiling"""
        from ydata_profiling import ProfileReport
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Настройка в зависимости от режима
                config = {
                    "minimal": self.mode == "minimal",
                    "explorative": self.mode == "explorative",
                    "title": self.title,
                    "sensitive": bool(self.sensitive_features)
                }
                
                # Если есть чувствительные признаки, скрываем их
                df_to_profile = container.data.copy()
                if self.sensitive_features:
                    for col in self.sensitive_features:
                        if col in df_to_profile.columns:
                            df_to_profile[col] = "[REDACTED]"
                
                self._profile = ProfileReport(df_to_profile, **config)
            
            # Извлекаем структурированные данные
            if hasattr(self._profile, 'get_description'):
                desc = self._profile.get_description()
                container.profile = self._extract_ydata_profile(desc, container.data)
            else:
                container.profile = self._create_basic_profile(container.data)
            
        except Exception as e:
            # Fallback to fast profiling
            if self.verbose:
                self._log(f"ydata-profiling failed: {e}. Using fast profiling.")
            container.profile = self._create_basic_profile(container.data)
        
        return container
    
    def _profile_fast(self, container: DataContainer) -> DataContainer:
        """Быстрое профилирование (без ydata-profiling)"""
        container.profile = self._create_basic_profile(container.data)
        return container
    
    def _extract_ydata_profile(self, desc: Any, original_df: pd.DataFrame) -> dict:
        """Извлечение данных из ydata-profiling"""
        profile = {
            "summary": {},
            "variables": {},
            "correlations": {},
            "missing": {},
            "alerts": [],
            "interactions": {},
            "samples": {}
        }
        
        try:
            # Основные метрики
            profile["summary"] = {
                "n_rows": desc.table.get("n", len(original_df)),
                "n_cols": desc.table.get("n_var", len(original_df.columns)),
                "missing_cells": desc.table.get("n_cells_missing", 0),
                "missing_percent": desc.table.get("p_cells_missing", 0) * 100,
                "duplicate_rows": desc.table.get("n_duplicates", 0),
                "duplicate_percent": desc.table.get("p_duplicates", 0) * 100,
                "memory_size": desc.table.get("memory_size", 0),
                "record_size": desc.table.get("record_size", 0),
                "types_distribution": desc.table.get("types", {})
            }
            
            # Информация по переменным
            for var_name, var_info in desc.variables.items():
                if var_name not in original_df.columns:
                    continue
                    
                var_data = original_df[var_name]
                var_profile = {
                    "type": str(var_info.get("type", "Unknown")),
                    "missing": {
                        "count": var_info.get("n_missing", 0),
                        "percent": var_info.get("p_missing", 0) * 100
                    },
                    "unique": {
                        "count": var_info.get("n_unique", 0),
                        "percent": var_info.get("p_unique", 0) * 100 if var_info.get("p_unique") else 0
                    },
                    "infinite": {
                        "count": var_info.get("n_infinite", 0),
                        "percent": var_info.get("p_infinite", 0) * 100 if var_info.get("p_infinite") else 0
                    }
                }
                
                # Числовые статистики
                if "mean" in var_info:
                    var_profile["numeric_stats"] = {
                        "mean": var_info.get("mean"),
                        "std": var_info.get("std"),
                        "variance": var_info.get("variance"),
                        "min": var_info.get("min"),
                        "max": var_info.get("max"),
                        "range": var_info.get("range"),
                        "5%": var_info.get("5%"),
                        "25%": var_info.get("25%"),
                        "50%": var_info.get("50%", var_info.get("median")),
                        "75%": var_info.get("75%"),
                        "95%": var_info.get("95%"),
                        "iqr": var_info.get("iqr"),
                        "cv": var_info.get("cv"),
                        "skewness": var_info.get("skewness"),
                        "kurtosis": var_info.get("kurtosis")
                    }
                
                # Категориальные статистики
                elif "value_counts" in var_info:
                    var_profile["categorical_stats"] = {
                        "top": var_info.get("value_counts_without_nan", {}),
                        "n_categories": len(var_info.get("value_counts_without_nan", {})),
                        "chi_squared": var_info.get("chi_squared")
                    }
                
                profile["variables"][var_name] = var_profile
            
            # Корреляции
            if hasattr(desc, 'correlations'):
                for corr_name, corr_matrix in desc.correlations.items():
                    if corr_matrix is not None:
                        profile["correlations"][corr_name] = corr_matrix.to_dict()
            
            # Алерты
            if hasattr(desc, 'alerts'):
                profile["alerts"] = [
                    {
                        "column": alert.column_name,
                        "type": alert.alert_type.name,
                        "description": str(alert),
                        "priority": self._get_alert_priority(alert.alert_type.name)
                    }
                    for alert in desc.alerts
                ]
            
            # Взаимодействия (если включены)
            if self.include_interactions and hasattr(desc, 'interactions'):
                profile["interactions"] = desc.interactions
            
            # Примеры данных
            profile["samples"] = {
                "head": original_df.head(10).to_dict(orient="records"),
                "tail": original_df.tail(10).to_dict(orient="records"),
                "sample": original_df.sample(min(10, len(original_df))).to_dict(orient="records")
            }
            
        except Exception as e:
            profile["_error"] = f"Error extracting ydata profile: {str(e)[:200]}"
            profile.update(self._create_basic_profile(original_df))
        
        return profile
    
    def _create_basic_profile(self, df: pd.DataFrame) -> dict:
        """Создание базового профиля"""
        profile = {
            "summary": self._get_summary_stats(df),
            "variables": {},
            "correlations": {},
            "missing": self._analyze_missing_patterns(df),
            "alerts": [],
            "samples": {
                "head": df.head(5).to_dict(orient="records"),
                "sample": df.sample(min(5, len(df))).to_dict(orient="records")
            }
        }
        
        # Анализ каждой колонки
        for col in df.columns:
            series = df[col]
            var_profile = self._analyze_variable(series, col)
            profile["variables"][col] = var_profile
            
            # Проверяем на проблемы
            alerts = self._check_variable_problems(series, col)
            profile["alerts"].extend(alerts)
        
        # Корреляции (только для числовых)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            profile["correlations"]["pearson"] = corr_matrix.to_dict()
            
            # Находим высокие корреляции
            high_corr = self._find_high_correlations(corr_matrix)
            if high_corr:
                profile["alerts"].extend(high_corr)
        
        return profile
    
    def _get_summary_stats(self, df: pd.DataFrame) -> dict:
        """Основные статистики датасета"""
        return {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "missing_cells": df.isnull().sum().sum(),
            "missing_percent": df.isnull().mean().mean() * 100,
            "duplicate_rows": df.duplicated().sum(),
            "duplicate_percent": df.duplicated().mean() * 100,
            "memory_size": df.memory_usage(deep=True).sum(),
            "data_types": df.dtypes.value_counts().to_dict(),
            "date_range": self._get_date_range(df) if any(pd.api.types.is_datetime64_any_dtype(df[col]) for col in df.columns) else None
        }
    
    def _analyze_variable(self, series: pd.Series, col_name: str) -> dict:
        """Анализ одной переменной"""
        analysis = {
            "type": str(series.dtype),
            "missing": {
                "count": series.isnull().sum(),
                "percent": series.isnull().mean() * 100
            },
            "unique": {
                "count": series.nunique(),
                "percent": series.nunique() / max(1, len(series)) * 100
            }
        }
        
        # Числовые статистики
        if pd.api.types.is_numeric_dtype(series):
            numeric_stats = series.describe().to_dict()
            numeric_stats.update({
                "skewness": series.skew(),
                "kurtosis": series.kurtosis(),
                "zeros": (series == 0).sum(),
                "zeros_percent": (series == 0).mean() * 100,
                "outliers": self._detect_outliers_iqr(series)
            })
            analysis["numeric_stats"] = numeric_stats
        
        # Категориальные статистики
        elif series.dtype == 'object' or series.dtype.name == 'category':
            value_counts = series.value_counts()
            analysis["categorical_stats"] = {
                "top_values": value_counts.head(10).to_dict(),
                "n_categories": len(value_counts),
                "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                "most_common_percent": (value_counts.iloc[0] / len(series) * 100) if len(value_counts) > 0 else 0
            }
        
        # Временные ряды
        elif pd.api.types.is_datetime64_any_dtype(series):
            analysis["datetime_stats"] = {
                "min": series.min(),
                "max": series.max(),
                "range": (series.max() - series.min()).days if not series.isnull().all() else None,
                "has_future_dates": (series > pd.Timestamp.now()).any() if not series.isnull().all() else False
            }
        
        return analysis
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> dict:
        """Анализ паттернов пропусков"""
        missing_matrix = df.isnull()
        
        return {
            "total": missing_matrix.sum().sum(),
            "by_column": missing_matrix.sum().to_dict(),
            "by_row": missing_matrix.sum(axis=1).value_counts().to_dict(),
            "patterns": self._find_missing_patterns(missing_matrix),
            "complete_cases": (~missing_matrix.any(axis=1)).sum(),
            "complete_cases_percent": (~missing_matrix.any(axis=1)).mean() * 100
        }
    
    def _find_missing_patterns(self, missing_matrix: pd.DataFrame) -> list:
        """Поиск паттернов пропусков"""
        patterns = []
        unique_patterns = missing_matrix.drop_duplicates()
        
        for pattern in unique_patterns.itertuples(index=False):
            pattern_tuple = tuple(pattern)
            count = (missing_matrix == pattern_tuple).all(axis=1).sum()
            percent = count / len(missing_matrix) * 100
            
            if percent > 1:  # Показываем только значимые паттерны (>1%)
                patterns.append({
                    "pattern": pattern_tuple,
                    "columns_missing": [col for col, is_missing in zip(missing_matrix.columns, pattern) if is_missing],
                    "count": count,
                    "percent": percent
                })
        
        return sorted(patterns, key=lambda x: x["percent"], reverse=True)[:10]
    
    def _detect_outliers_iqr(self, series: pd.Series) -> dict:
        """Обнаружение выбросов методом IQR"""
        series_clean = series.dropna()
        
        if len(series_clean) < 10:
            return {"count": 0, "percent": 0}
        
        q1 = series_clean.quantile(0.25)
        q3 = series_clean.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            return {"count": 0, "percent": 0}
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        outliers = (series_clean < lower) | (series_clean > upper)
        count = outliers.sum()
        
        return {
            "count": int(count),
            "percent": count / len(series_clean) * 100,
            "lower_bound": float(lower),
            "upper_bound": float(upper)
        }
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame) -> list:
        """Поиск высоких корреляций"""
        alerts = []
        corr_abs = corr_matrix.abs()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr = corr_abs.loc[col1, col2]
                
                if corr > self.correlation_threshold:
                    alerts.append({
                        "column": f"{col1}, {col2}",
                        "type": "HIGH_CORRELATION",
                        "description": f"High correlation ({corr:.3f}) between {col1} and {col2}",
                        "priority": "medium",
                        "details": {"correlation": float(corr)}
                    })
        
        return alerts
    
    def _check_variable_problems(self, series: pd.Series, col_name: str) -> list:
        """Проверка переменной на проблемы"""
        alerts = []
        
        # Высокий % пропусков
        missing_percent = series.isnull().mean() * 100
        if missing_percent > self.missing_threshold * 100:
            alerts.append({
                "column": col_name,
                "type": "HIGH_MISSING",
                "description": f"{missing_percent:.1f}% missing values",
                "priority": "high" if missing_percent > 80 else "medium"
            })
        
        # Константное значение
        if series.nunique() == 1:
            alerts.append({
                "column": col_name,
                "type": "CONSTANT",
                "description": "Constant value",
                "priority": "high"
            })
        
        # Почти константное
        elif series.nunique() == 2:
            value_counts = series.value_counts(normalize=True)
            if max(value_counts) > 0.95:
                alerts.append({
                    "column": col_name,
                    "type": "NEARLY_CONSTANT",
                    "description": f"Nearly constant: {max(value_counts)*100:.1f}% one value",
                    "priority": "medium"
                })
        
        # Высокая уникальность (возможно ID)
        unique_percent = series.nunique() / len(series) * 100
        if unique_percent > 95 and series.dtype in ['object', 'string']:
            alerts.append({
                "column": col_name,
                "type": "HIGH_UNIQUENESS",
                "description": f"High uniqueness ({unique_percent:.1f}%), possibly ID column",
                "priority": "low"
            })
        
        return alerts
    
    def _get_alert_priority(self, alert_type: str) -> str:
        """Определение приоритета алерта"""
        high_priority = ["CONSTANT", "HIGH_MISSING", "DUPLICATES"]
        medium_priority = ["NEARLY_CONSTANT", "HIGH_CORRELATION", "SKEWED", "HIGH_CARDINALITY"]
        
        if alert_type in high_priority:
            return "high"
        elif alert_type in medium_priority:
            return "medium"
        else:
            return "low"
    
    def _add_recommendations(self, container: DataContainer):
        """Добавление рекомендаций на основе профиля"""
        if not container.profile or "alerts" not in container.profile:
            return
        
        # Группируем алерты по приоритету
        alerts_by_priority = {"high": [], "medium": [], "low": []}
        
        for alert in container.profile["alerts"]:
            priority = alert.get("priority", "low")
            alerts_by_priority[priority].append(alert)
        
        # Добавляем рекомендации
        for priority in ["high", "medium", "low"]:
            if alerts_by_priority[priority]:
                container.recommendations.append({
                    "type": f"profile_{priority}_alerts",
                    "count": len(alerts_by_priority[priority]),
                    "alerts": alerts_by_priority[priority][:5],  # Первые 5
                    "message": f"Found {len(alerts_by_priority[priority])} {priority} priority issues"
                })
    
    def get_report(self) -> Any:
        """Получить объект отчёта"""
        return self._profile
    
    def save_report(self, path: str) -> None:
        """Сохранить HTML отчёт"""
        if self._profile is not None and hasattr(self._profile, 'to_file'):
            self._profile.to_file(path)
        else:
            # Генерируем простой HTML отчёт
            self._generate_simple_report(path)
    
    def _generate_simple_report(self, path: str):
        """Генерация простого HTML отчёта"""
        if not self._profile_dict:
            return
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Profile Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1, h2, h3 { color: #333; }
                .alert { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .alert-high { background: #ffebee; border-left: 4px solid #f44336; }
                .alert-medium { background: #fff3e0; border-left: 4px solid #ff9800; }
                .alert-low { background: #e8f5e9; border-left: 4px solid #4caf50; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background: #f5f5f5; }
                .summary { background: #f0f8ff; padding: 20px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>📊 Data Profile Report</h1>
            <div class="summary">
                <h2>Dataset Summary</h2>
                <p><strong>Rows:</strong> {n_rows:,} | <strong>Columns:</strong> {n_cols}</p>
                <p><strong>Missing:</strong> {missing_percent:.1f}% | <strong>Duplicates:</strong> {duplicate_percent:.1f}%</p>
            </div>
            
            <h2>⚠️ Issues Found</h2>
            {alerts_html}
            
            <h2>📈 Recommendations</h2>
            <ul>
                <li>Install ydata-profiling for detailed reports: <code>pip install ydata-profiling</code></li>
                <li>Review high priority issues above</li>
                <li>Consider imputation for missing values</li>
            </ul>
            
            <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
                <p>Generated by ML Data Forge | {timestamp}</p>
            </footer>
        </body>
        </html>
        """
        
        # Заполняем шаблон
        from datetime import datetime
        
        alerts_html = ""
        if "alerts" in self._profile_dict:
            for alert in self._profile_dict["alerts"][:10]:  # Первые 10 алертов
                priority = alert.get("priority", "low")
                alerts_html += f"""
                <div class="alert alert-{priority}">
                    <strong>{alert.get('type', 'ALERT')}</strong>: {alert.get('description', '')}
                    <br><small>Column: {alert.get('column', 'N/A')}</small>
                </div>
                """
        
        html = html.format(
            n_rows=self._profile_dict.get("summary", {}).get("n_rows", 0),
            n_cols=self._profile_dict.get("summary", {}).get("n_cols", 0),
            missing_percent=self._profile_dict.get("summary", {}).get("missing_percent", 0),
            duplicate_percent=self._profile_dict.get("summary", {}).get("duplicate_percent", 0),
            alerts_html=alerts_html or "<p>No major issues found.</p>",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)