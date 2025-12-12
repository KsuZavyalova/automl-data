# automl_data/adapters/image/__init__.py
"""
Адаптеры для обработки изображений.
"""

from .preprocessor import ImagePreprocessor
from .augmentor import ImageAugmentor

__all__ = ["ImagePreprocessor", "ImageAugmentor"]