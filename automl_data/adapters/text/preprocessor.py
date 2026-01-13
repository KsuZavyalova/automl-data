# automl_data/adapters/text/preprocessor.py
"""
Препроцессор текста (только английский язык).

Два уровня:
- minimal: базовая очистка для трансформеров (BERT, RoBERTa, GPT)
- full: полная предобработка для классических методов (TF-IDF, Word2Vec)
"""

from __future__ import annotations

import re
import html
import unicodedata
import logging
from typing import List, Optional

import pandas as pd

from ..base import BaseAdapter
from ...core.container import DataContainer, ProcessingStage
from ...core.config import TextConfig
from ...utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


class TextPreprocessor(BaseAdapter):
    """
    Препроцессор текста для английского языка.
    
    Parameters
    ----------
    config : TextConfig, optional
        Конфигурация предобработки
    preprocessing_level : str
        Уровень предобработки: "minimal" или "full"
    
    Example
    -------
    >>> preprocessor = TextPreprocessor(preprocessing_level="minimal")
    >>> container = preprocessor.fit_transform(container)
    
    >>> # Для классических методов
    >>> preprocessor = TextPreprocessor(preprocessing_level="full")
    >>> container = preprocessor.fit_transform(container)
    """
    
    # Паттерны для очистки
    HTML_PATTERN = re.compile(r'<[^>]+>')
    URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    PUNCTUATION_PATTERN = re.compile(r'[^\w\s]')
    NUMBERS_PATTERN = re.compile(r'\d+')
    
    ENGLISH_STOPWORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
        "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
        'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
        'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
        'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
        'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
        'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
        'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
        've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
        'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
        'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
        "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
        "weren't", 'won', "won't", 'wouldn', "wouldn't"
    }
    
    def __init__(
        self,
        config: TextConfig | None = None,
        preprocessing_level: str = "minimal",
        **kwargs
    ):
        super().__init__(name="TextPreprocessor", **kwargs)
        
        self.config = config or TextConfig(preprocessing_level=preprocessing_level)
        self.preprocessing_level = self.config.preprocessing_level
        
        self._lemmatizer = None
        self._nltk_initialized = False
    
    def _init_nltk(self) -> None:
        """Ленивая инициализация NLTK"""
        if self._nltk_initialized:
            return
        
        try:
            import nltk
            
            for resource in ['wordnet', 'averaged_perceptron_tagger', 'punkt']:
                try:
                    nltk.data.find(f'corpora/{resource}' if resource == 'wordnet' 
                                   else f'taggers/{resource}' if 'tagger' in resource 
                                   else f'tokenizers/{resource}')
                except LookupError:
                    nltk.download(resource, quiet=True)
            
            from nltk.stem import WordNetLemmatizer
            self._lemmatizer = WordNetLemmatizer()
            self._nltk_initialized = True
            
        except ImportError:
            logger.warning("NLTK not available. Lemmatization will be skipped.")
            self._nltk_initialized = True
    
    def _detect_language(self, text: str) -> str:
        """
        Определение языка текста.
        
        Returns
        -------
        str
            Код языка ('en' для английского)
        
        Raises
        ------
        ValidationError
            Если текст не на английском языке
        """
        # Простая эвристика: проверяем наличие английских слов
        words = text.lower().split()[:50]  # Берём первые 50 слов
        
        english_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'then', 'once', 'and',
            'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither', 'not',
            'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also'
        }
        
        english_count = sum(1 for w in words if w in english_words)
        english_ratio = english_count / max(len(words), 1)
        
        if english_ratio < 0.1:
            return "non-english"
        
        return "en"
    
    def _validate_language(self, texts: pd.Series) -> None:
        """Проверка, что все тексты на английском языке"""
        sample_size = min(100, len(texts))
        sample = texts.dropna().sample(n=sample_size, random_state=42) if len(texts) > sample_size else texts.dropna()
        
        non_english_count = 0
        for text in sample:
            if self._detect_language(str(text)) != "en":
                non_english_count += 1
        
        non_english_ratio = non_english_count / max(len(sample), 1)
        
        if non_english_ratio > 0.3:
            raise ValidationError(
                f"Dataset contains {non_english_ratio:.1%} non-English texts. "
                "This module only supports English language. "
                "Please filter your dataset to include only English texts."
            )
    
    def _fit_impl(self, container: DataContainer) -> None:
        """Анализ текстов"""
        if not container.text_column:
            return
        
        texts = container.data[container.text_column].dropna()
        
        self._validate_language(texts)
        
        lengths = texts.str.len()
        word_counts = texts.str.split().str.len()
        
        self._fit_info = {
            "total_texts": len(texts),
            "avg_length": lengths.mean(),
            "avg_words": word_counts.mean(),
            "preprocessing_level": self.preprocessing_level,
            "language": "en"
        }
        
        logger.info(
            f"TextPreprocessor fitted: {len(texts)} texts, "
            f"avg {word_counts.mean():.0f} words, "
            f"level={self.preprocessing_level}"
        )
    
    def _transform_impl(self, container: DataContainer) -> DataContainer:
        """Применение предобработки"""
        if not container.text_column:
            return container
        
        df = container.data.copy()
        text_col = container.text_column
        
        original_count = len(df)
        
        df[text_col] = df[text_col].apply(self._preprocess_text)
        
        df = df[df[text_col].str.len() >= self.config.min_text_length]
        df = df[df[text_col].str.len() <= self.config.max_text_length]
        df = df[df[text_col].str.strip().str.len() > 0]
        
        df = df.reset_index(drop=True)
        
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} texts (empty or wrong length)")
        
        container.data = df
        container.stage = ProcessingStage.CLEANED
        
        container.recommendations.append({
            "type": "preprocessing",
            "level": self.preprocessing_level,
            "original_count": original_count,
            "filtered_count": filtered_count,
            "final_count": len(df)
        })
        
        return container
    
    def _preprocess_text(self, text: str) -> str:
        """
        Предобработка одного текста.
        
        Минимальная обработка (для трансформеров):
        - Удаление HTML
        - Удаление URL
        - Удаление email
        - Нормализация Unicode
        - Нормализация пробелов
        
        Полная обработка (для классических методов):
        - Всё вышеперечисленное
        - Приведение к нижнему регистру
        - Удаление пунктуации
        - Удаление чисел (опционально)
        - Удаление стоп-слов
        - Лемматизация
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # === МИНИМАЛЬНАЯ ОБРАБОТКА ===
        
        if self.config.remove_html:
            text = html.unescape(text)
            text = self.HTML_PATTERN.sub(' ', text)
        
        if self.config.remove_urls:
            text = self.URL_PATTERN.sub(' ', text)
        
        if self.config.remove_emails:
            text = self.EMAIL_PATTERN.sub(' ', text)
        
        if self.config.fix_unicode:
            text = unicodedata.normalize('NFKC', text)
            text = ''.join(char for char in text if char.isprintable() or char in '\n\t ')
        
        if self.config.normalize_whitespace:
            text = self.WHITESPACE_PATTERN.sub(' ', text)
            text = text.strip()
        
        if self.preprocessing_level == "minimal":
            return text
        
        # === ПОЛНАЯ ОБРАБОТКА ===
        
        if self.config.lowercase:
            text = text.lower()
        
        if self.config.remove_numbers:
            text = self.NUMBERS_PATTERN.sub(' ', text)
        
        if self.config.remove_punctuation:
            text = self.PUNCTUATION_PATTERN.sub(' ', text)
        
        words = text.split()
        
        if self.config.remove_stopwords:
            words = [w for w in words if w.lower() not in self.ENGLISH_STOPWORDS]

        if self.config.lemmatize:
            words = self._lemmatize_words(words)
        
        text = ' '.join(words)
        
        text = self.WHITESPACE_PATTERN.sub(' ', text).strip()
        
        return text
    
    def _lemmatize_words(self, words: List[str]) -> List[str]:
        """Лемматизация слов с помощью NLTK WordNet"""
        self._init_nltk()
        
        if self._lemmatizer is None:
            return words
        
        try:
            from nltk import pos_tag
            from nltk.corpus import wordnet
            
            def get_wordnet_pos(tag: str) -> str:
                """Конвертация POS тега в формат WordNet"""
                if tag.startswith('J'):
                    return wordnet.ADJ
                elif tag.startswith('V'):
                    return wordnet.VERB
                elif tag.startswith('N'):
                    return wordnet.NOUN
                elif tag.startswith('R'):
                    return wordnet.ADV
                return wordnet.NOUN
            
            tagged = pos_tag(words)
            
            lemmatized = [
                self._lemmatizer.lemmatize(word, get_wordnet_pos(tag))
                for word, tag in tagged
            ]
            
            return lemmatized
            
        except Exception as e:
            logger.warning(f"Lemmatization failed: {e}")
            return words
    
    def preprocess_single(self, text: str) -> str:
        """
        Предобработка одного текста (для использования вне пайплайна).
        
        Parameters
        ----------
        text : str
            Входной текст
        
        Returns
        -------
        str
            Обработанный текст
        """
        return self._preprocess_text(text)