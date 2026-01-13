# automl_data/core/config.py
"""
Конфигурация AutoForge для разных типов данных.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path


class TaskType(Enum):
    """Типы задач машинного обучения"""
    AUTO = "auto"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class TabularConfig:
    """Конфигурация для табличных данных"""
    impute_strategy: str = "auto"
    scaling: str = "auto"
    encode_strategy: str = "auto"
    max_onehot_cardinality: int = 10
    outlier_method: str = "auto"
    outlier_action: str = "clip"


@dataclass
class TextConfig:
    """
    Конфигурация для текстовой обработки (только английский язык).
    
    Уровни предобработки:
    - "minimal": Только базовая очистка (для трансформеров: BERT, RoBERTa, etc.)
    - "full": Полная предобработка (для классических ML: TF-IDF, Word2Vec, etc.)
    """
    
    # Уровень предобработки
    preprocessing_level: str = "full"  # "minimal" или "full"
    
    # === Минимальная предобработка (для трансформеров) ===
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    normalize_whitespace: bool = True
    fix_unicode: bool = True
    
    # === Полная предобработка (для классических ML) ===
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_stopwords: bool = True
    lemmatize: bool = True
    
    # Фильтрация по длине
    min_text_length: int = 3
    max_text_length: int = 10000
    
    # === Аугментация ===
    augment: bool = False
    augment_factor: float = 2.0  # Во сколько раз увеличить датасет
    
    augment_methods: List[str] = field(default_factory=lambda: [
        "eda",
        "synonym_wordnet",
        "pronoun_to_noun",
        "t5_paraphrase"
    ])
    
    # EDA параметры
    eda_alpha_sr: float = 0.1  # Synonym replacement ratio
    eda_alpha_ri: float = 0.1  # Random insertion ratio
    eda_alpha_rs: float = 0.1  # Random swap ratio
    eda_alpha_rd: float = 0.1  # Random deletion ratio
    eda_num_aug: int = 4       # Количество аугментаций на текст
    
    # T5 параметры
    t5_model_name: str = "t5-small"
    t5_num_beams: int = 4
    t5_max_length: int = 256
    
    balance_classes: bool = False  # Балансировать через аугментацию
    imbalance_threshold: float = 0.3  # Порог дисбаланса (min/max ratio)
    
    def __post_init__(self):
        """Валидация конфигурации"""
        valid_levels = {"minimal", "full"}
        if self.preprocessing_level not in valid_levels:
            raise ValueError(
                f"preprocessing_level must be one of {valid_levels}, "
                f"got '{self.preprocessing_level}'"
            )
        
        valid_methods = {"eda", "t5_paraphrase", "synonym_wordnet", "pronoun_to_noun"}
        for method in self.augment_methods:
            if method not in valid_methods:
                raise ValueError(
                    f"Unknown augmentation method: '{method}'. "
                    f"Valid methods: {valid_methods}"
                )
        
        self.augment_methods = list(dict.fromkeys(self.augment_methods))
        
        if self.augment_factor < 1.0:
            self.augment_factor = 1.0


@dataclass
class ImageConfig:
    """Конфигурация для изображений"""
    target_size: tuple[int, int] = (224, 224)
    keep_aspect_ratio: bool = True
    
    normalize: bool = True
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    
    augment: bool = True
    augment_factor: float = 3.0
    
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_range: int = 15
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    zoom_range: tuple[float, float] = (0.9, 1.1)
    
    use_randaugment: bool = True
    use_mixup: bool = False
    use_cutmix: bool = False
    use_cutout: bool = True
    
    add_noise: bool = True
    add_blur: bool = True
    
    shift_range: float = 0.1
    shear_range: float = 0.1
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    hue_range: float = 0.1
    
    output_dir: Optional[Path] = None
    examples_dir: Optional[Path] = None
    
    random_state: int = 42
    verbose: bool = True
    save_examples: bool = True
    n_examples: int = 10
    
    balance_classes: bool = False
    
    def __post_init__(self):
        if self.output_dir:
            self.output_dir = Path(self.output_dir)
        if self.examples_dir:
            self.examples_dir = Path(self.examples_dir)


@dataclass
class ForgeConfig:
    """Полная конфигурация AutoForge"""
    
    target: str | None = None
    task: TaskType = TaskType.AUTO
    
    tabular: TabularConfig = field(default_factory=TabularConfig)
    text: TextConfig = field(default_factory=TextConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    
    balance: bool = True
    balance_threshold: float = 0.3
    
    test_size: float = 0.2
    stratify: bool = True
    
    random_state: int = 42
    n_jobs: int = -1
    verbose: bool = True
    
    def __post_init__(self):
        if isinstance(self.task, str):
            self.task = TaskType(self.task) if self.task != "auto" else TaskType.AUTO
    
    def to_dict(self) -> dict[str, Any]:
        """Конвертировать конфиг в словарь"""
        return {
            "target": self.target,
            "task": self.task.value,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "balance": self.balance,
            "balance_threshold": self.balance_threshold,
            "tabular_config": {
                "impute_strategy": self.tabular.impute_strategy,
                "scaling": self.tabular.scaling,
                "encode_strategy": self.tabular.encode_strategy,
                "max_onehot_cardinality": self.tabular.max_onehot_cardinality,
                "outlier_method": self.tabular.outlier_method,
                "outlier_action": self.tabular.outlier_action,
            },
            "text_config": {
                "preprocessing_level": self.text.preprocessing_level,
                "remove_html": self.text.remove_html,
                "remove_urls": self.text.remove_urls,
                "remove_emails": self.text.remove_emails,
                "normalize_whitespace": self.text.normalize_whitespace,
                "fix_unicode": self.text.fix_unicode,
                "lowercase": self.text.lowercase,
                "remove_punctuation": self.text.remove_punctuation,
                "remove_numbers": self.text.remove_numbers,
                "remove_stopwords": self.text.remove_stopwords,
                "lemmatize": self.text.lemmatize,
                "min_text_length": self.text.min_text_length,
                "max_text_length": self.text.max_text_length,
                "augment": self.text.augment,
                "augment_factor": self.text.augment_factor,
                "augment_methods": self.text.augment_methods,
                "balance_classes": self.text.balance_classes,
                "imbalance_threshold": self.text.imbalance_threshold,
            },
            "image_config": {
                "target_size": self.image.target_size,
                "keep_aspect_ratio": self.image.keep_aspect_ratio,
                "normalize": self.image.normalize,
                "augment": self.image.augment,
                "augment_factor": self.image.augment_factor,
                "balance_classes": self.image.balance_classes,
            }
        }