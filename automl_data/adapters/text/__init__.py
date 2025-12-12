# automl_data/adapters/text/__init__.py
"""
Текстовые адаптеры (только английский язык).
"""

from .preprocessor import TextPreprocessor
from .augmentor import TextAugmentor

__all__ = [
    "TextPreprocessor",
    "TextAugmentor"
]