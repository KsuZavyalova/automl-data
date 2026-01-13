# automl_data/core/forge.py
"""
AutoForge — главный класс библиотеки для автоматической подготовки данных.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from .container import DataContainer, DataType
from .pipeline import Pipeline
from .config import ForgeConfig, TaskType, TextConfig, ImageConfig, TabularConfig

from ..adapters.profiling import ProfilerAdapter
from ..adapters.feature_cleaner import FeatureCleanerAdapter
from ..adapters.encoding import EncodingAdapter
from ..adapters.outliers import OutlierAdapter
from ..adapters.balancing import BalancingAdapter
from ..adapters.imputation import ImputationAdapter
from ..adapters.scaling import ScalingAdapter


from ..utils.decorators import timing, require_fitted
from ..utils.exceptions import ValidationError

@dataclass
class ForgeResult:
    """
    Результат обработки данных.
    
    Предоставляет удобный доступ к обработанным данным,
    сплитам и отчётам.
    
    Attributes:
        container: Обработанный DataContainer
        config: Использованная конфигурация
        execution_time: Время обработки в секундах
        profile_report: Объект отчёта ydata-profiling (если доступен)
    
    Example:
        >>> result = forge.fit_transform(df)
        >>> X_train, X_test, y_train, y_test = result.get_splits()
        >>> result.save_report("report.html")
    """
    
    container: DataContainer
    config: ForgeConfig
    execution_time: float = 0.0
    profile_report: Any = None
    
    @property
    def data(self) -> pd.DataFrame:
        """Обработанный DataFrame"""
        return self.container.data
    
    @property
    def X(self) -> pd.DataFrame:
        """Признаки (без target)"""
        return self.container.X
    
    @property
    def y(self) -> pd.Series | None:
        """Целевая переменная"""
        return self.container.y
    
    @property
    def quality_score(self) -> float:
        """Оценка качества данных (0-1)"""
        return self.container.quality_score
    
    @property
    def shape(self) -> tuple[int, int]:
        """Размерность данных"""
        return self.container.shape
    
    @property
    def steps(self) -> list[str]:
        """Список выполненных шагов"""
        return [s.name for s in self.container.processing_history]
    
    @property
    def profile(self) -> dict[str, Any]:
        """Профиль данных"""
        return self.container.profile
    
    @property
    def recommendations(self) -> list[dict[str, Any]]:
        """Рекомендации от адаптеров"""
        return self.container.recommendations
    
    def get_splits(
        self,
        test_size: float | None = None,
        random_state: int | None = None,
        stratify: bool | None = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Получить train/test сплиты.
        
        Args:
            test_size: Размер тестовой выборки (по умолчанию из конфига)
            random_state: Random seed (по умолчанию из конфига)
            stratify: Использовать стратификацию (по умолчанию из конфига)
        
        Returns:
            (X_train, X_test, y_train, y_test)
        
        Example:
            >>> X_train, X_test, y_train, y_test = result.get_splits()
            >>> model.fit(X_train, y_train)
        """
        from sklearn.model_selection import train_test_split
        
        test_size = test_size or self.config.test_size
        random_state = random_state or self.config.random_state
        stratify_flag = stratify if stratify is not None else self.config.stratify
        
        if self.y is None:
            raise ValueError("Target column not specified or not found")
        
        # Стратификация для классификации
        strat = None
        if stratify_flag:
            # Проверяем, подходит ли target для стратификации
            if self.y.nunique() < 50 and self.y.value_counts().min() >= 2:
                strat = self.y
        
        return train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=strat
        )
    
    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray | None]:
        """
        Конвертация в numpy arrays.
        
        Returns:
            (X, y) как numpy arrays
        """
        X = self.X.values
        y = self.y.values if self.y is not None else None
        return X, y
    
    def save_report(self, path: str | Path) -> None:
        """
        Сохранить HTML отчёт.
        
        Args:
            path: Путь для сохранения
        """
        path = Path(path)
        
        if self.profile_report is not None and hasattr(self.profile_report, 'to_file'):
            self.profile_report.to_file(str(path))

    
    def get_pipeline_code(self) -> str:
        """Получить воспроизводимый Python код"""
        return self.container.get_pipeline_code() if hasattr(self.container, 'get_pipeline_code') else ""
    
    def summary(self) -> dict[str, Any]:
        """Полная сводка о результате"""
        return {
            **self.container.summary(),
            "execution_time": f"{self.execution_time:.2f}s",
            "steps": self.steps,
            "config": self.config.to_dict()
        }
    
    def __repr__(self) -> str:
        return (
            f"ForgeResult(shape={self.shape}, quality={self.quality_score:.0%}, "
            f"steps={len(self.steps)}, time={self.execution_time:.2f}s)"
        )


class AutoForge:
    """
    Автоматическая подготовка данных для машинного обучения.
    
    AutoForge анализирует данные и автоматически:
    - Определяет тип данных (табличные, текст, изображения)
    - Выбирает оптимальные стратегии обработки
    - Применяет соответствующий пайплайн
    - Балансирует классы при необходимости
    
    Example:
        >>> # Простое использование
        >>> forge = AutoForge(target="price")
        >>> result = forge.fit_transform(df)
        >>> X_train, X_test, y_train, y_test = result.get_splits()
        
        >>> # С настройками
        >>> forge = AutoForge(
        ...     target="sentiment",
        ...     text_column="review",
        ...     balance=True,
        ...     verbose=True
        ... )
        >>> result = forge.fit_transform(df)
    
    Parameters
    ----------
    target : str, optional
        Название целевой колонки
    task : str, default="auto"
        Тип задачи: "classification", "regression", "auto"
    text_column : str, optional
        Колонка с текстом (для NLP задач)
    image_column : str, optional
        Колонка с путями к изображениям
    image_dir : str, optional
        Директория с изображениями
    impute_strategy : str, default="auto"
        Стратегия заполнения пропусков
    scaling : str, default="auto"
        Метод масштабирования
    encode_strategy : str, default="auto"
        Стратегия кодирования категорий
    outlier_method : str, default="auto"
        Метод обнаружения выбросов
    balance : bool, default=True
        Балансировать ли классы
    test_size : float, default=0.2
        Размер тестовой выборки
    random_state : int, default=42
        Seed для воспроизводимости
    verbose : bool, default=True
        Выводить ли логи
    """
    
    def __init__(
        self,
        target: str | None = None,
        task: str = "auto",
        
        # Для текста
        text_column: str | None = None,
        
        # Для изображений
        image_column: str | None = None,
        image_dir: str | Path | None = None,
        
        # Табличные настройки
        impute_strategy: str = "auto",
        scaling: str = "auto",
        encode_strategy: str = "auto",
        max_onehot_cardinality: int = 10,
        outlier_method: str = "auto",
        outlier_action: str = "clip",
        
        # Балансировка
        balance: bool = True,
        balance_strategy: str = "auto",
        balance_threshold: float = 0.3,
        
        # Разбиение
        test_size: float = 0.2,
        stratify: bool = True,
        
        # Общее
        random_state: int = 42,
        verbose: bool = True,
        
        text_config: TextConfig | None = None,
        image_config: ImageConfig | None = None,
        tabular_config: TabularConfig | None = None,

        # Текстовые настройки
        text_preprocessing_level: str = "minimal",
        text_remove_html: bool = True,
        text_remove_urls: bool = True,
        text_remove_emails: bool = True,
        text_normalize_whitespace: bool = True,
        text_fix_unicode: bool = True,
        text_lowercase: bool = True,
        text_remove_punctuation: bool = True,
        text_remove_numbers: bool = False,
        text_remove_stopwords: bool = True,
        text_lemmatize: bool = True,
        text_min_length: int = 3,
        text_max_length: int = 10000,
        text_augment: bool = False,
        text_augment_factor: float = 2.0,
        text_augment_methods: list[str] | None = None,
        text_balance_classes: bool = False,

        # Настройки изображений
        augment: bool | None = None,
        augment_factor: float | None = None,
        target_size: tuple[int, int] | None = None,
        keep_aspect_ratio: bool | None = None,
        normalize: bool | None = None,
        horizontal_flip: bool | None = None,
        rotation_range: int | None = None,
        brightness_range: tuple[float, float] | None = None,
        contrast_range: tuple[float, float] | None = None,
        zoom_range: tuple[float, float] | None = None,
        use_randaugment: bool | None = None,
        
        **kwargs
    ):
        if tabular_config is None:
            tabular_config = TabularConfig(
                impute_strategy=impute_strategy,
                scaling=scaling,
                encode_strategy=encode_strategy,
                max_onehot_cardinality=max_onehot_cardinality,
                outlier_method=outlier_method,
                outlier_action=outlier_action
            )
        
        if text_config is None:
            text_config = TextConfig(
                preprocessing_level=text_preprocessing_level,
                remove_html=text_remove_html,
                remove_urls=text_remove_urls,
                remove_emails=text_remove_emails,
                normalize_whitespace=text_normalize_whitespace,
                fix_unicode=text_fix_unicode,
                lowercase=text_lowercase,
                remove_punctuation=text_remove_punctuation,
                remove_numbers=text_remove_numbers,
                remove_stopwords=text_remove_stopwords,
                lemmatize=text_lemmatize,
                min_text_length=text_min_length,
                max_text_length=text_max_length,
                augment=text_augment,
                augment_factor=text_augment_factor,
                augment_methods=text_augment_methods or [
                    "eda",
                    "synonym_wordnet",
                    "pronoun_to_noun"
                ],
                balance_classes=text_balance_classes,
            )
        
        if image_config is None:
            image_config = ImageConfig()
        
            if augment is not None:
                image_config.augment = augment
            if augment_factor is not None:
                image_config.augment_factor = augment_factor
            if target_size is not None:
                image_config.target_size = target_size
            if keep_aspect_ratio is not None:
                image_config.keep_aspect_ratio = keep_aspect_ratio
            if normalize is not None:
                image_config.normalize = normalize
            if horizontal_flip is not None:
                image_config.horizontal_flip = horizontal_flip
            if rotation_range is not None:
                image_config.rotation_range = rotation_range
            if brightness_range is not None:
                image_config.brightness_range = brightness_range
            if contrast_range is not None:
                image_config.contrast_range = contrast_range
            if zoom_range is not None:
                image_config.zoom_range = zoom_range
            if use_randaugment is not None:
                image_config.use_randaugment = use_randaugment
        
        self.config = ForgeConfig(
            target=target,
            task=TaskType(task) if task != "auto" else TaskType.AUTO,
            tabular=tabular_config,
            text=text_config,
            image=image_config,
            balance=balance,
            balance_threshold=balance_threshold,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
            verbose=verbose,
            **kwargs
        )
        
        self.text_column = text_column
        self.image_column = image_column
        self.image_dir = Path(image_dir) if image_dir else None
        self.balance_strategy = balance_strategy

        self._logger = logging.getLogger("automl_data.AutoForge")
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s | %(message)s',
                datefmt='%H:%M:%S'
            )

        self._pipeline: Pipeline | None = None
        self._profiler: ProfilerAdapter | None = None
        self._is_fitted = False
        self._data_type: DataType = DataType.TABULAR
    
    def fit(self, data: pd.DataFrame | DataContainer) -> AutoForge:
        """
        Анализ данных и настройка пайплайна.
                
        Returns:
            self (для цепочки вызовов)
        """
        if isinstance(data, pd.DataFrame):
            container = DataContainer(
                data=data.copy(),
                target_column=self.config.target,
                text_column=self.text_column,
                image_column=self.image_column,
                image_dir=self.image_dir,
                imbalance_threshold=self.config.balance_threshold  # ПЕРЕДАЕМ ПОРОГ!
            )
        else:
            container = data.clone()
            container.imbalance_threshold = self.config.balance_threshold  # ОБНОВЛЯЕМ
            if self.config.target:
                container.target_column = self.config.target
        
        self._log(f"🔍 Analyzing data: {container.shape[0]:,} rows × {container.shape[1]} columns")
        
        self._validate_input(container)
        
        self._data_type = container.data_type
        self._log(f"📋 Data type: {self._data_type.name}")
        
        if self.config.task == TaskType.AUTO:
            self.config.task = self._infer_task(container)
            self._log(f"📋 Detected task: {self.config.task.value}")
        
        if self.config.verbose:
            try:
                self._profiler = ProfilerAdapter(minimal=True)
                container = self._profiler.transform(container)
                
                missing = container.profile.get("missing_percent", 0)
                alerts = len(container.profile.get("alerts", []))
                self._log(f"📊 Profile: {missing:.1f}% missing, {alerts} alerts")
            except Exception:
                pass
        
        self._pipeline = self._build_pipeline(container)
        self._log(f"Pipeline ready with {len(self._pipeline)} steps")
        
        self._is_fitted = True
        return self
    
    @timing
    @require_fitted
    def transform(self, data: pd.DataFrame | DataContainer) -> ForgeResult:
        """
        Применение пайплайна к данным.
        """

        if isinstance(data, pd.DataFrame):
            container = DataContainer(
                data=data.copy(),
                target_column=self.config.target,
                text_column=self.text_column,
                image_column=self.image_column,
                image_dir=self.image_dir
            )
        else:
            container = data.clone()
    
        self._log("Transforming data...")
        result = self._pipeline.execute(container)        
        if not result.success and result.errors:
            self._log(f"Pipeline completed with errors: {result.errors}")
        quality_score = self._calculate_quality(result.container)
        result.container.quality_score = quality_score
        execution_time = getattr(self.transform, 'last_execution_time', 0.0)
        self._log(f"Done! Shape: {result.container.shape}, Quality: {quality_score:.0%}, Time: {execution_time:.2f}s")
        
        return ForgeResult(
            container=result.container,
            config=self.config,
            execution_time=execution_time,
            profile_report=self._profiler.get_report() if self._profiler else None
        )
    
    def fit_transform(self, data: pd.DataFrame | DataContainer) -> ForgeResult:
        """
        Анализ и обработка данных в одном вызове.
        Это основной метод для большинства случаев.
        """
        return self.fit(data).transform(data)
    
    def _validate_input(self, container: DataContainer) -> None:
        """Валидация входных данных"""
        if len(container) == 0:
            raise ValidationError("Data is empty")
        
        if self.config.target:
            if self.config.target not in container.columns:
                available = container.columns[:5]
                raise ValidationError(
                    f"Target column '{self.config.target}' not found. "
                    f"Available: {available}..."
                )
    
    def _infer_task(self, container: DataContainer) -> TaskType:
        """Определение типа задачи по данным"""

        target = container.data[self.config.target]

        if pd.api.types.is_numeric_dtype(target):
            unique_ratio = target.nunique() / len(target)
            if unique_ratio > 0.1 or target.nunique() > 20:
                return TaskType.REGRESSION
        
        return TaskType.CLASSIFICATION 
    
    def _build_pipeline(self, container: DataContainer) -> Pipeline:
        """Построение пайплайна на основе типа данных"""
        pipeline = Pipeline(name="AutoForge", verbose=self.config.verbose)
        
        if container.is_tabular:
            return self._build_tabular_pipeline(pipeline, container)
        elif container.is_text:
            return self._build_text_pipeline(pipeline, container)
        elif container.is_image:
            return self._build_image_pipeline(pipeline, container)
        else:
            return self._build_tabular_pipeline(pipeline, container)
    
    def _build_tabular_pipeline(self, pipeline: Pipeline, container: DataContainer) -> Pipeline:
        """Пайплайн для табличных данных"""
        cfg = self.config.tabular

        pipeline.add_step(
            FeatureCleanerAdapter(
                max_missing_ratio=0.9,
                remove_duplicates=True,
                correlation_threshold=0.95
            ),
            name="FeatureCleaning",
            on_error="warn"
        )
        
        pipeline.add_step(
            ImputationAdapter(
                strategy=cfg.impute_strategy,
                numeric_strategy="median",
                categorical_strategy="most_frequent"
            ),
            name="Imputation",
            on_error="warn"
        )
        
        if cfg.outlier_method != "none":
            pipeline.add_step(
                OutlierAdapter(
                    method=cfg.outlier_method,
                    action=cfg.outlier_action
                ),
                name="Outliers",
                on_error="warn"
            )
        
        if container.categorical_columns:
            pipeline.add_step(
                EncodingAdapter(
                    strategy=cfg.encode_strategy,
                    target_column=self.config.target,
                    max_onehot_cardinality=cfg.max_onehot_cardinality
                ),
                name="Encoding",
                on_error="warn"
            )
        
        if cfg.scaling != "none":
            pipeline.add_step(
                ScalingAdapter(
                    strategy=cfg.scaling
                ),
                name="Scaling",
                on_error="warn"
            )
        
        if self.config.balance and self.config.task == TaskType.CLASSIFICATION:
            pipeline.add_step(
                BalancingAdapter(
                    strategy=self.balance_strategy,
                    target_column=self.config.target,
                    imbalance_threshold=self.config.balance_threshold,
                    random_state=self.config.random_state
                ),
                name="Balancing",
                on_error="warn"
            )
        
        return pipeline
    
    def _build_text_pipeline(self, pipeline: Pipeline, container: DataContainer) -> Pipeline:
        """Пайплайн для текстовых данных"""
        from ..adapters.text import TextPreprocessor, TextAugmentor
        
        cfg = self.config.text
        
        pipeline.add_step(
            TextPreprocessor(
                config=cfg,
                preprocessing_level=cfg.preprocessing_level
            ),
            name="TextPreprocessing"
        )
        
        augment_needed = (
            cfg.augment or 
            (self.config.balance and container.is_imbalanced) or
            cfg.augment_factor > 1.0
        )
        
        if augment_needed:
            pipeline.add_step(
                TextAugmentor(
                    config=cfg,
                    augment_factor=cfg.augment_factor,
                    balance_classes=self.config.balance and container.is_imbalanced,
                    method_priority=cfg.augment_methods,
                    random_state=self.config.random_state
                ),
                name="TextAugmentation",
                on_error="warn"
            )
        
        return pipeline
    
    def _build_image_pipeline(self, pipeline: Pipeline, container: DataContainer) -> Pipeline:
        """Пайплайн для изображений"""
        from ..adapters.image import ImagePreprocessor, ImageAugmentor
        
        cfg = self.config.image
        
        pipeline.add_step(
            ImagePreprocessor(config=cfg),
            name="ImagePreprocessing"
        )
        
        # Условия для аугментации:
        # A) Пользователь явно запросил аугментацию (cfg.augment=True)
        # B) Нужна балансировка через аугментацию
        
        explicit_augment = cfg.augment and cfg.augment_factor > 1.0
        balance_augment = (
            self.config.balance and 
            container.is_imbalanced and 
            getattr(cfg, 'balance_classes', False)
        )
        
        if explicit_augment or balance_augment:
            self._log(f"Adding augmentation: explicit={explicit_augment}, balance={balance_augment}")
            
            pipeline.add_step(
                ImageAugmentor(
                    config=cfg,
                    output_dir=self.image_dir,
                    verbose=self.config.verbose,
                    random_state=self.config.random_state
                ),
                name="ImageAugmentation",
                on_error="warn"
            )
        else:
            if cfg.augment:
                self._log(f"Augmentation requested but skipped: "
                        f"augment_factor={cfg.augment_factor}, "
                        f"balance={self.config.balance}, "
                        f"is_imbalanced={container.is_imbalanced}")
        
        return pipeline

    
    def _calculate_quality(self, container: DataContainer) -> float:
        """Расчёт качества данных"""
        df = container.data
        completeness = 1 - df.isnull().mean().mean()
        try:
            uniqueness = 1 - (df.duplicated().sum() / max(1, len(df)))
        except TypeError:
            # Если есть numpy массивы, используем фиксированное значение
            uniqueness = 0.9
        
        # 3. Консистентность типов
        numeric_ratio = len(container.numeric_columns) / max(1, len(container.columns))
        
        balance_score = 1.0
        if container.class_distribution and len(container.class_distribution) >= 2:
            counts = list(container.class_distribution.values())
            balance_score = min(counts) / max(counts)
        
        score = (
            0.35 * completeness +
            0.25 * uniqueness +
            0.20 * numeric_ratio +
            0.20 * balance_score
        )
        
        return min(1.0, max(0.0, score))
    
    def _log(self, message: str) -> None:
        """Логирование"""
        if self.config.verbose:
            self._logger.info(message)
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"AutoForge(target='{self.config.target}', task={self.config.task.value}, {status})"