# automl_data/utils/text_utils.py
"""
Утилиты для работы с текстом.
"""

import re
from typing import Literal, Optional


def detect_language(text: str, threshold: float = 0.6) -> Literal["ru", "en", "unknown"]:
    """
    Определяет язык текста.
    
    Args:
        text: Текст для определения
        threshold: Порог уверенности (0.6 = 60%)
    
    Returns:
        "ru", "en" или "unknown"
    """
    if not text or len(text.strip()) < 10:
        return "unknown"
    
    text = text.lower()
    
    # Русские символы
    ru_pattern = r'[а-яё]'
    ru_count = len(re.findall(ru_pattern, text))
    ru_ratio = ru_count / max(1, len(text))
    
    # Английские символы
    en_pattern = r'[a-z]'
    en_count = len(re.findall(en_pattern, text))
    en_ratio = en_count / max(1, len(text))
    
    # Определяем язык
    if ru_ratio > threshold and ru_ratio > en_ratio:
        return "ru"
    elif en_ratio > threshold and en_ratio > ru_ratio:
        return "en"
    else:
        return "unknown"


def validate_language(text: str, expected_lang: str) -> bool:
    """
    Проверяет, соответствует ли текст ожидаемому языку.
    
    Args:
        text: Текст для проверки
        expected_lang: Ожидаемый язык ("ru" или "en")
    
    Returns:
        True если соответствует
    """
    detected = detect_language(text)
    return detected == expected_lang or detected == "unknown"