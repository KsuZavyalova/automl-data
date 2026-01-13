# automl_data/adapters/text/augmentor.py
"""
Аугментация текста (только английский язык).

Методы:
- EDA (Easy Data Augmentation): synonym replacement, random swap, delete, insert
- T5 Paraphrase: перефразирование через T5
- Synonym WordNet: замена слов синонимами из WordNet
- Pronoun to Noun: замена местоимений на существительные
"""

from __future__ import annotations

import random
import logging
from typing import List, Dict, Optional, Callable, Any, Set
from collections import Counter

import pandas as pd
import numpy as np

from ..base import BaseAdapter
from ...core.container import DataContainer, ProcessingStage
from ...core.config import TextConfig
from ...utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


class TextAugmentor(BaseAdapter):
    """
    Аугментатор текста для английского языка.
    
    Поддерживает два режима:
    1. Увеличение объёма данных (augment_factor > 1)
    2. Балансировка классов (balance_classes=True)
    
    Parameters
    ----------
    config : TextConfig, optional
        Конфигурация аугментации
    augment_factor : float
        Во сколько раз увеличить датасет
    balance_classes : bool
        Балансировать классы через аугментацию
    random_state : int
        Seed для воспроизводимости
    
    Example
    -------
    >>> # Увеличение объёма
    >>> augmentor = TextAugmentor(augment_factor=2.0)
    >>> container = augmentor.fit_transform(container)
    
    >>> # Балансировка классов
    >>> augmentor = TextAugmentor(balance_classes=True)
    >>> container = augmentor.fit_transform(container)
    """
    
    STOPWORDS: Set[str] = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
        'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'can', 'will', 'should',
        'now'
    }
    
    PRONOUN_REPLACEMENTS: Dict[str, List[str]] = {
        'he': ['the man', 'the person', 'the individual', 'the guy'],
        'she': ['the woman', 'the person', 'the individual', 'the lady'],
        'it': ['the thing', 'the object', 'the item'],
        'they': ['the people', 'the group', 'the individuals', 'the team'],
        'him': ['the man', 'the person', 'the individual'],
        'her': ['the woman', 'the person', 'the individual'],
        'them': ['the people', 'the group', 'the individuals'],
        'his': ["the man's", "the person's"],
        'hers': ["the woman's", "the person's"],
        'their': ["the people's", "the group's"],
        'someone': ['a person', 'an individual', 'somebody'],
        'anyone': ['any person', 'anybody'],
        'everyone': ['all people', 'everybody', 'all individuals'],
    }
    
    def __init__(
        self,
        config: TextConfig | None = None,
        augment_factor: float = 2.0,
        balance_classes: bool = False,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(name="TextAugmentor", **kwargs)
        
        self.config = config or TextConfig()
        self.augment_factor = augment_factor if augment_factor > 1.0 else self.config.augment_factor
        self.balance_classes = balance_classes or self.config.balance_classes
        self.random_state = random_state
        
        random.seed(random_state)
        np.random.seed(random_state)
        
        self._wordnet_initialized = False
        self._t5_model = None
        self._t5_tokenizer = None
        
        self._methods: Dict[str, Callable[[str], str]] = {}
        self._available_methods: List[str] = []
    
    def _init_wordnet(self) -> bool:
        """Инициализация WordNet"""
        if self._wordnet_initialized:
            return True
        
        try:
            import nltk
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            
            from nltk.corpus import wordnet
            self._wordnet = wordnet
            self._wordnet_initialized = True
            logger.info("WordNet initialized successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to initialize WordNet: {e}")
            return False
    
    def _init_t5(self) -> bool:
        """Инициализация T5 модели для перефразирования"""
        if self._t5_model is not None:
            return True
        
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            import torch
            
            model_name = self.config.t5_model_name
            logger.info(f"Loading T5 model: {model_name}")
            
            self._t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
            self._t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
            
            # Переносим на GPU если доступен
            if torch.cuda.is_available():
                self._t5_model = self._t5_model.cuda()
            
            self._t5_model.eval()
            logger.info("T5 model loaded successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to initialize T5: {e}")
            return False
    
    def _init_methods(self) -> None:
        """Инициализация доступных методов аугментации"""
        self._methods = {}
        self._available_methods = []
        
        for method_name in self.config.augment_methods:
            if method_name == "eda":
                # EDA всегда доступен
                self._methods["eda"] = self._augment_eda
                self._available_methods.append("eda")
                
            elif method_name == "synonym_wordnet":
                if self._init_wordnet():
                    self._methods["synonym_wordnet"] = self._augment_synonym_wordnet
                    self._available_methods.append("synonym_wordnet")
                    
            elif method_name == "t5_paraphrase":
                if self._init_t5():
                    self._methods["t5_paraphrase"] = self._augment_t5_paraphrase
                    self._available_methods.append("t5_paraphrase")
                    
            elif method_name == "pronoun_to_noun":
                # Всегда доступен
                self._methods["pronoun_to_noun"] = self._augment_pronoun_to_noun
                self._available_methods.append("pronoun_to_noun")
        
        if not self._available_methods:
            # Fallback на базовый EDA
            self._methods["eda"] = self._augment_eda
            self._available_methods.append("eda")
        
        logger.info(f"Available augmentation methods: {self._available_methods}")
    
    def _fit_impl(self, container: DataContainer) -> None:
        """Анализ данных для аугментации"""
        if not container.text_column:
            return
        
        self._init_methods()
        
        df = container.data
        target_col = container.target_column
        
        class_counts = {}
        if target_col and target_col in df.columns:
            class_counts = df[target_col].value_counts().to_dict()
        
        self._fit_info = {
            "n_samples": len(df),
            "class_distribution": class_counts,
            "available_methods": self._available_methods,
            "augment_factor": self.augment_factor,
            "balance_classes": self.balance_classes
        }
        
        logger.info(
            f"TextAugmentor fitted: {len(df)} samples, "
            f"methods={self._available_methods}, "
            f"balance={self.balance_classes}, "
            f"factor={self.augment_factor}"
        )
    
    def _transform_impl(self, container: DataContainer) -> DataContainer:
        """Применение аугментации"""
        if not container.text_column:
            return container
        
        df = container.data.copy()
        text_col = container.text_column
        target_col = container.target_column
        
        original_size = len(df)
        
        if self.balance_classes and target_col and target_col in df.columns:
            # Режим балансировки классов
            augmented_df = self._balance_via_augmentation(df, text_col, target_col)
        else:
            # Режим увеличения объёма
            augmented_df = self._augment_dataset(df, text_col)
        
        container.data = augmented_df.reset_index(drop=True)
        container.stage = ProcessingStage.AUGMENTED
        
        new_size = len(container.data)
        logger.info(f"Augmentation complete: {original_size} -> {new_size} samples")
        
        container.recommendations.append({
            "type": "augmentation",
            "original_size": original_size,
            "new_size": new_size,
            "methods_used": self._available_methods,
            "balance_mode": self.balance_classes
        })
        
        return container
    
    def _balance_via_augmentation(
        self,
        df: pd.DataFrame,
        text_col: str,
        target_col: str
    ) -> pd.DataFrame:
        """
        Балансировка классов через аугментацию миноритарных классов.
        """
        class_counts = df[target_col].value_counts()
        max_count = class_counts.max()
        
        logger.info(f"Class distribution before balancing: {class_counts.to_dict()}")
        
        augmented_dfs = [df]
        
        for class_label, count in class_counts.items():
            if count >= max_count:
                continue
            
            # Сколько нужно добавить
            needed = max_count - count
            class_df = df[df[target_col] == class_label]
            
            logger.info(f"Augmenting class '{class_label}': {count} -> {max_count} (+{needed})")
            
            # Генерируем аугментации
            augmented_texts = []
            augmented_rows = []
            
            # Повторяем пока не наберём нужное количество
            while len(augmented_texts) < needed:
                for _, row in class_df.iterrows():
                    if len(augmented_texts) >= needed:
                        break
                    
                    text = row[text_col]
                    aug_text = self._augment_single(text)
                    
                    if aug_text and aug_text != text:
                        new_row = row.copy()
                        new_row[text_col] = aug_text
                        augmented_rows.append(new_row)
                        augmented_texts.append(aug_text)
            
            if augmented_rows:
                aug_df = pd.DataFrame(augmented_rows)
                augmented_dfs.append(aug_df)
        
        result = pd.concat(augmented_dfs, ignore_index=True)
        
        new_counts = result[target_col].value_counts()
        logger.info(f"Class distribution after balancing: {new_counts.to_dict()}")
        
        return result
    
    def _augment_dataset(
        self,
        df: pd.DataFrame,
        text_col: str
    ) -> pd.DataFrame:
        """
        Увеличение объёма датасета в augment_factor раз.
        """
        if self.augment_factor <= 1.0:
            return df
        
        target_size = int(len(df) * self.augment_factor)
        needed = target_size - len(df)
        
        logger.info(f"Augmenting dataset: {len(df)} -> {target_size} (+{needed})")
        
        augmented_rows = []
        
        # Перемешиваем индексы для разнообразия
        indices = df.index.tolist()
        random.shuffle(indices)
        
        idx_cycle = 0
        while len(augmented_rows) < needed:
            idx = indices[idx_cycle % len(indices)]
            row = df.loc[idx]
            
            text = row[text_col]
            aug_text = self._augment_single(text)
            
            if aug_text and aug_text != text:
                new_row = row.copy()
                new_row[text_col] = aug_text
                augmented_rows.append(new_row)
            
            idx_cycle += 1
            
            if idx_cycle > len(indices) * 10:
                logger.warning("Augmentation cycle limit reached")
                break
        
        if augmented_rows:
            aug_df = pd.DataFrame(augmented_rows)
            return pd.concat([df, aug_df], ignore_index=True)
        
        return df
    
    def _augment_single(self, text: str) -> str:
        """Аугментация одного текста случайным методом"""
        if not text or len(text.split()) < 3:
            return text
        
        method_name = random.choice(self._available_methods)
        method = self._methods[method_name]
        
        try:
            return method(text)
        except Exception as e:
            logger.debug(f"Augmentation failed with {method_name}: {e}")
            return text
    
    
    def _augment_eda(self, text: str) -> str:
        """
        Easy Data Augmentation (EDA).
        
        Случайно применяет один из методов:
        - Synonym Replacement
        - Random Insertion
        - Random Swap
        - Random Deletion
        """
        words = text.split()
        if len(words) < 3:
            return text
        
        methods = [
            (self._eda_synonym_replacement, self.config.eda_alpha_sr),
            (self._eda_random_insertion, self.config.eda_alpha_ri),
            (self._eda_random_swap, self.config.eda_alpha_rs),
            (self._eda_random_deletion, self.config.eda_alpha_rd),
        ]
        
        method, alpha = random.choice(methods)
        n = max(1, int(len(words) * alpha))
        
        augmented_words = method(words, n)
        return ' '.join(augmented_words)
    
    def _eda_synonym_replacement(self, words: List[str], n: int) -> List[str]:
        """Замена n случайных слов синонимами"""
        new_words = words.copy()
        
        replaceable = [i for i, w in enumerate(words) 
                      if w.lower() not in self.STOPWORDS and w.isalpha()]
        
        random.shuffle(replaceable)
        
        replaced = 0
        for idx in replaceable[:n]:
            word = words[idx]
            synonym = self._get_synonym(word)
            if synonym:
                new_words[idx] = synonym
                replaced += 1
        
        return new_words
    
    def _eda_random_insertion(self, words: List[str], n: int) -> List[str]:
        """Вставка n случайных синонимов"""
        new_words = words.copy()
        
        for _ in range(n):
            candidates = [w for w in words if w.lower() not in self.STOPWORDS and w.isalpha()]
            if not candidates:
                break
            
            word = random.choice(candidates)
            synonym = self._get_synonym(word)
            
            if synonym:
                insert_pos = random.randint(0, len(new_words))
                new_words.insert(insert_pos, synonym)
        
        return new_words
    
    def _eda_random_swap(self, words: List[str], n: int) -> List[str]:
        """Случайная перестановка n пар слов"""
        new_words = words.copy()
        
        for _ in range(n):
            if len(new_words) < 2:
                break
            
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return new_words
    
    def _eda_random_deletion(self, words: List[str], n: int) -> List[str]:
        """Случайное удаление n слов"""
        if len(words) <= n + 1:
            return words
        
        new_words = words.copy()
        
        delete_indices = random.sample(range(len(new_words)), min(n, len(new_words) - 1))
        delete_indices.sort(reverse=True)
        
        for idx in delete_indices:
            del new_words[idx]
        
        return new_words
    
    def _get_synonym(self, word: str) -> Optional[str]:
        """Получение синонима из WordNet"""
        if not self._wordnet_initialized:
            self._init_wordnet()
        
        if not self._wordnet_initialized:
            return None
        
        try:
            synsets = self._wordnet.synsets(word)
            if not synsets:
                return None
            
            synonyms = set()
            for syn in synsets[:3]: 
                for lemma in syn.lemmas()[:5]:  
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        synonyms.add(synonym)
            
            if synonyms:
                return random.choice(list(synonyms))
            
            return None
            
        except Exception:
            return None
    
    def _augment_synonym_wordnet(self, text: str) -> str:
        """Замена нескольких слов синонимами из WordNet"""
        words = text.split()
        if len(words) < 3:
            return text
        
        n = max(1, int(len(words) * 0.2))  # Заменяем 20% слов
        new_words = self._eda_synonym_replacement(words, n)
        
        return ' '.join(new_words)
    
    
    def _augment_t5_paraphrase(self, text: str) -> str:
        """Перефразирование через T5"""
        if self._t5_model is None:
            return text
        
        try:
            import torch
            
            # Формируем промпт
            input_text = f"paraphrase: {text}"
            
            inputs = self._t5_tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=self.config.t5_max_length,
                truncation=True
            )
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            with torch.no_grad():
                outputs = self._t5_model.generate(
                    inputs,
                    max_length=self.config.t5_max_length,
                    num_beams=self.config.t5_num_beams,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
            
            paraphrase = self._t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if paraphrase and paraphrase.lower() != text.lower():
                return paraphrase
            
            return text
            
        except Exception as e:
            logger.debug(f"T5 paraphrase failed: {e}")
            return text
    
    
    def _augment_pronoun_to_noun(self, text: str) -> str:
        """Замена местоимений на существительные"""
        words = text.split()
        new_words = []
        
        changed = False
        for word in words:
            lower_word = word.lower()
            
            if lower_word in self.PRONOUN_REPLACEMENTS:
                replacement = random.choice(self.PRONOUN_REPLACEMENTS[lower_word])
                
                if word[0].isupper():
                    replacement = replacement.capitalize()
                
                new_words.append(replacement)
                changed = True
            else:
                new_words.append(word)
        
        if changed:
            return ' '.join(new_words)
        
        return text
    
    def augment_single(self, text: str, method: str | None = None) -> str:
        """
        Аугментация одного текста (для использования вне пайплайна).
        
        Parameters
        ----------
        text : str
            Входной текст
        method : str, optional
            Метод аугментации. Если не указан — выбирается случайно.
        
        Returns
        -------
        str
            Аугментированный текст
        """
        if not self._methods:
            self._init_methods()
        
        if method and method in self._methods:
            return self._methods[method](text)
        
        return self._augment_single(text)
    
    def augment_batch(
        self,
        texts: List[str],
        n_augmentations: int = 1
    ) -> List[str]:
        """
        Аугментация списка текстов.
        
        Parameters
        ----------
        texts : List[str]
            Список текстов
        n_augmentations : int
            Количество аугментаций на каждый текст
        
        Returns
        -------
        List[str]
            Список аугментированных текстов
        """
        if not self._methods:
            self._init_methods()
        
        results = []
        for text in texts:
            for _ in range(n_augmentations):
                aug = self._augment_single(text)
                results.append(aug)
        
        return results