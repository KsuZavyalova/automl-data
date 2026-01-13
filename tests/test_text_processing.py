# tests/test_text_processing.py
"""
Юнит-тесты для TextPreprocessor и TextAugmentor.

Тестирует:
- TextPreprocessor: очистка, нормализация, лемматизация
- TextAugmentor: EDA, синонимы, балансировка
- Интеграция с DataContainer
- Edge cases
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from automl_data.core.container import DataContainer
from automl_data.core.config import TextConfig
from automl_data.adapters.text.preprocessor import TextPreprocessor
from automl_data.adapters.text.augmentor import TextAugmentor
from automl_data.utils.exceptions import ValidationError


# ==================== FIXTURES ====================

@pytest.fixture
def simple_text_df() -> pd.DataFrame:
    """Простой текстовый датасет"""
    return pd.DataFrame({
        'text': [
            'This is a simple test sentence.',
            'Another example of English text here.',
            'The quick brown fox jumps over the lazy dog.',
            'Machine learning is an interesting field.',
            'Natural language processing is useful.',
        ],
        'label': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def dirty_text_df() -> pd.DataFrame:
    """Датасет с грязным текстом"""
    return pd.DataFrame({
        'text': [
            '<p>This is <b>HTML</b> text</p>',
            'Check out https://example.com for more info!',
            'Contact us at test@email.com today',
            'Too    many     spaces   here',
            'Special chars: @#$%^&*()',
            'Numbers like 12345 and more text',
            'UPPERCASE AND lowercase MiXeD',
        ],
        'label': [0, 1, 0, 1, 0, 1, 0]
    })


@pytest.fixture
def imbalanced_text_df() -> pd.DataFrame:
    """Несбалансированный текстовый датасет"""
    texts_class_0 = [
        f'This is a positive review number {i}. Great product!'
        for i in range(20)
    ]
    texts_class_1 = [
        f'This is a negative review number {i}. Bad experience.'
        for i in range(5)
    ]
    
    return pd.DataFrame({
        'text': texts_class_0 + texts_class_1,
        'label': [0] * 20 + [1] * 5
    })


@pytest.fixture
def long_text_df() -> pd.DataFrame:
    """Датасет с длинными текстами"""
    long_text = "This is a long sentence with many words. " * 50
    short_text = "Short."
    normal_text = "This is a normal length sentence for testing purposes."
    
    return pd.DataFrame({
        'text': [long_text, short_text, normal_text, normal_text, normal_text],
        'label': [0, 1, 0, 1, 0]
    })


@pytest.fixture
def multiclass_text_df() -> pd.DataFrame:
    """Мультиклассовый текстовый датасет"""
    return pd.DataFrame({
        'text': [
            'Sports news about football match.',
            'Political debate in congress today.',
            'Technology advances in AI field.',
            'Sports team wins championship.',
            'Political election results announced.',
            'New technology gadget released.',
        ] * 5,
        'category': ['sports', 'politics', 'tech', 'sports', 'politics', 'tech'] * 5
    })


def make_text_container(df: pd.DataFrame, text_col: str = 'text', target_col: str = 'label') -> DataContainer:
    """Хелпер для создания контейнера"""
    return DataContainer(
        data=df.copy(),
        text_column=text_col,
        target_column=target_col
    )


# ==================== TEXT PREPROCESSOR TESTS ====================

class TestTextPreprocessorBasic:
    """Базовые тесты TextPreprocessor"""
    
    def test_initialization_default(self):
        """Тест инициализации по умолчанию"""
        preprocessor = TextPreprocessor()
        
        assert preprocessor.preprocessing_level == "minimal"
        assert preprocessor.config is not None
    
    def test_initialization_minimal(self):
        """Тест инициализации с minimal уровнем"""
        preprocessor = TextPreprocessor(preprocessing_level="minimal")
        
        assert preprocessor.preprocessing_level == "minimal"
    
    def test_initialization_full(self):
        """Тест инициализации с full уровнем"""
        preprocessor = TextPreprocessor(preprocessing_level="full")
        
        assert preprocessor.preprocessing_level == "full"
    
    def test_initialization_with_config(self):
        """Тест инициализации с TextConfig"""
        config = TextConfig(
            preprocessing_level="full",
            remove_stopwords=True,
            lemmatize=True
        )
        
        preprocessor = TextPreprocessor(config=config)
        
        assert preprocessor.preprocessing_level == "full"
        assert preprocessor.config.remove_stopwords is True
    
    def test_fit_transform_basic(self, simple_text_df):
        """Базовый тест fit_transform"""
        container = make_text_container(simple_text_df)
        
        preprocessor = TextPreprocessor(preprocessing_level="minimal")
        result = preprocessor.fit_transform(container)
        
        assert preprocessor.is_fitted
        assert len(result.data) > 0
        assert result.data['text'].notna().all()
    
    def test_fit_info_populated(self, simple_text_df):
        """Проверка заполнения fit_info"""
        container = make_text_container(simple_text_df)
        
        preprocessor = TextPreprocessor()
        preprocessor.fit(container)
        
        assert "total_texts" in preprocessor._fit_info
        assert "avg_length" in preprocessor._fit_info
        assert "avg_words" in preprocessor._fit_info
        assert preprocessor._fit_info["language"] == "en"


class TestTextPreprocessorMinimal:
    """Тесты минимальной предобработки"""
    
    def test_removes_html(self, dirty_text_df):
        """Тест удаления HTML"""
        container = make_text_container(dirty_text_df)
        
        preprocessor = TextPreprocessor(preprocessing_level="minimal")
        result = preprocessor.fit_transform(container)
        
        # HTML теги должны быть удалены
        html_text = result.data['text'].iloc[0]
        assert '<p>' not in html_text
        assert '<b>' not in html_text
        assert '</p>' not in html_text
    
    def test_removes_urls(self, dirty_text_df):
        """Тест удаления URL"""
        container = make_text_container(dirty_text_df)
        
        preprocessor = TextPreprocessor(preprocessing_level="minimal")
        result = preprocessor.fit_transform(container)
        
        url_text = result.data['text'].iloc[1]
        assert 'https://' not in url_text
        assert 'example.com' not in url_text
    
    def test_removes_emails(self, dirty_text_df):
        """Тест удаления email"""
        container = make_text_container(dirty_text_df)
        
        preprocessor = TextPreprocessor(preprocessing_level="minimal")
        result = preprocessor.fit_transform(container)
        
        email_text = result.data['text'].iloc[2]
        assert '@' not in email_text
        assert 'email.com' not in email_text
    
    def test_normalizes_whitespace(self, dirty_text_df):
        """Тест нормализации пробелов"""
        container = make_text_container(dirty_text_df)
        
        preprocessor = TextPreprocessor(preprocessing_level="minimal")
        result = preprocessor.fit_transform(container)
        
        space_text = result.data['text'].iloc[3]
        assert '    ' not in space_text
        assert '  ' not in space_text
    
    def test_preserves_case_minimal(self, dirty_text_df):
        """Минимальная обработка сохраняет регистр"""
        container = make_text_container(dirty_text_df)
        
        preprocessor = TextPreprocessor(preprocessing_level="minimal")
        result = preprocessor.fit_transform(container)
        
        case_text = result.data['text'].iloc[6]
        # Должны остаться заглавные буквы
        assert any(c.isupper() for c in case_text)


class TestTextPreprocessorFull:
    """Тесты полной предобработки"""
    
    def test_lowercase_conversion(self, dirty_text_df):
        """Тест приведения к нижнему регистру"""
        container = make_text_container(dirty_text_df)
        
        preprocessor = TextPreprocessor(preprocessing_level="full")
        result = preprocessor.fit_transform(container)
        
        case_text = result.data['text'].iloc[6]
        # Не должно быть заглавных букв
        assert case_text == case_text.lower()
    
    def test_removes_punctuation(self, dirty_text_df):
        """Тест удаления пунктуации"""
        container = make_text_container(dirty_text_df)
        
        config = TextConfig(
            preprocessing_level="full",
            remove_punctuation=True
        )
        preprocessor = TextPreprocessor(config=config)
        result = preprocessor.fit_transform(container)
        
        punct_text = result.data['text'].iloc[4]
        # Спецсимволы должны быть удалены
        assert '@' not in punct_text
        assert '#' not in punct_text
        assert '%' not in punct_text
    
    def test_removes_stopwords(self, simple_text_df):
        """Тест удаления стоп-слов"""
        container = make_text_container(simple_text_df)
        
        config = TextConfig(
            preprocessing_level="full",
            remove_stopwords=True
        )
        preprocessor = TextPreprocessor(config=config)
        result = preprocessor.fit_transform(container)
        
        text = result.data['text'].iloc[0].lower()
        # Стоп-слова должны быть удалены
        assert ' is ' not in f' {text} '
        assert ' a ' not in f' {text} '
    
    def test_lemmatization(self, simple_text_df):
        """Тест лемматизации"""
        pytest.importorskip("nltk")
        
        df = pd.DataFrame({
            'text': [
                'The dogs are running quickly',
                'She was walking to the stores',
                'They were playing games yesterday',
            ],
            'label': [0, 1, 0]
        })
        
        container = make_text_container(df)
        
        config = TextConfig(
            preprocessing_level="full",
            lemmatize=True,
            remove_stopwords=False  # Чтобы не влияло
        )
        preprocessor = TextPreprocessor(config=config)
        result = preprocessor.fit_transform(container)
        
        # Проверяем, что лемматизация сработала
        # (результат зависит от NLTK, просто проверяем что не упало)
        assert len(result.data) > 0


class TestTextPreprocessorFiltering:
    """Тесты фильтрации текстов"""
    
    def test_filters_short_texts(self, long_text_df):
        """Тест фильтрации коротких текстов"""
        container = make_text_container(long_text_df)
        
        config = TextConfig(
            preprocessing_level="minimal",
            min_text_length=10
        )
        preprocessor = TextPreprocessor(config=config)
        result = preprocessor.fit_transform(container)
        
        # Короткий текст "Short." должен быть отфильтрован
        assert len(result.data) < len(long_text_df)
        assert all(len(t) >= 10 for t in result.data['text'])
    
    def test_filters_long_texts(self, long_text_df):
        """Тест фильтрации длинных текстов"""
        container = make_text_container(long_text_df)
        
        config = TextConfig(
            preprocessing_level="minimal",
            max_text_length=500
        )
        preprocessor = TextPreprocessor(config=config)
        result = preprocessor.fit_transform(container)
        
        # Очень длинный текст должен быть отфильтрован
        assert len(result.data) < len(long_text_df)
        assert all(len(t) <= 500 for t in result.data['text'])
    

class TestTextPreprocessorLanguage:
    """Тесты проверки языка"""
    
    def test_english_text_passes(self, simple_text_df):
        """Английский текст проходит валидацию"""
        container = make_text_container(simple_text_df)
        
        preprocessor = TextPreprocessor()
        
        # Не должно быть ошибки
        result = preprocessor.fit_transform(container)
        assert len(result.data) > 0
    
    def test_non_english_raises_error(self):
        """Не-английский текст вызывает ошибку"""
        df = pd.DataFrame({
            'text': [
                'Это русский текст',
                'Еще один пример на русском языке',
                'Третий текст тоже на русском',
            ] * 10,
            'label': [0, 1, 0] * 10
        })
        
        container = make_text_container(df)
        
        preprocessor = TextPreprocessor()
        
        with pytest.raises(ValidationError) as exc_info:
            preprocessor.fit_transform(container)
        
        assert "English" in str(exc_info.value) or "non-English" in str(exc_info.value).lower()


class TestTextPreprocessorSingleText:
    """Тесты обработки одиночного текста"""
    
    def test_preprocess_single_minimal(self):
        """Тест preprocess_single с minimal"""
        preprocessor = TextPreprocessor(preprocessing_level="minimal")
        
        text = "<p>Hello World!</p> Check https://test.com"
        result = preprocessor.preprocess_single(text)
        
        assert '<p>' not in result
        assert 'https://' not in result
        assert 'Hello' in result  # Регистр сохранён
    
    def test_preprocess_single_full(self):
        """Тест preprocess_single с full"""
        preprocessor = TextPreprocessor(preprocessing_level="full")
        
        text = "The Quick Brown Fox Jumps!"
        result = preprocessor.preprocess_single(text)
        
        assert result == result.lower()  # Нижний регистр
        assert '!' not in result  # Пунктуация удалена


# ==================== TEXT AUGMENTOR TESTS ====================

class TestTextAugmentorBasic:
    """Базовые тесты TextAugmentor"""
    
    def test_initialization_default(self):
        """Тест инициализации по умолчанию"""
        augmentor = TextAugmentor()
        
        assert augmentor.augment_factor == 2.0
        assert augmentor.balance_classes is False
    
    def test_initialization_with_params(self):
        """Тест инициализации с параметрами"""
        augmentor = TextAugmentor(
            augment_factor=3.0,
            balance_classes=True,
            random_state=123
        )
        
        assert augmentor.augment_factor == 3.0
        assert augmentor.balance_classes is True
        assert augmentor.random_state == 123
    
    def test_initialization_with_config(self):
        """Тест инициализации с TextConfig"""
        config = TextConfig(
            augment_factor=2.5,
            balance_classes=True
        )
        
        augmentor = TextAugmentor(config=config)
        
        # augment_factor берётся из параметра или config
        assert augmentor.balance_classes is True
    
    def test_fit_transform_basic(self, simple_text_df):
        """Базовый тест fit_transform"""
        container = make_text_container(simple_text_df)
        
        augmentor = TextAugmentor(augment_factor=2.0, random_state=42)
        result = augmentor.fit_transform(container)
        
        assert augmentor.is_fitted
        assert len(result.data) > len(simple_text_df)
    
    def test_fit_info_populated(self, simple_text_df):
        """Проверка заполнения fit_info"""
        container = make_text_container(simple_text_df)
        
        augmentor = TextAugmentor()
        augmentor.fit(container)
        
        assert "n_samples" in augmentor._fit_info
        assert "available_methods" in augmentor._fit_info
        assert "augment_factor" in augmentor._fit_info


class TestTextAugmentorAugmentation:
    """Тесты аугментации"""
    
    def test_augmentation_increases_size(self, simple_text_df):
        """Тест увеличения размера датасета"""
        container = make_text_container(simple_text_df)
        original_size = len(simple_text_df)
        
        augmentor = TextAugmentor(augment_factor=2.0, random_state=42)
        result = augmentor.fit_transform(container)
        
        # Размер должен увеличиться примерно в 2 раза
        expected_min = int(original_size * 1.5)
        assert len(result.data) >= expected_min
    
    def test_augmentation_preserves_labels(self, simple_text_df):
        """Тест сохранения меток"""
        container = make_text_container(simple_text_df)
        original_labels = set(simple_text_df['label'].unique())
        
        augmentor = TextAugmentor(augment_factor=2.0, random_state=42)
        result = augmentor.fit_transform(container)
        
        new_labels = set(result.data['label'].unique())
        
        assert new_labels == original_labels
    
    def test_augmented_texts_different(self, simple_text_df):
        """Тест, что аугментированные тексты отличаются"""
        container = make_text_container(simple_text_df)
        original_texts = set(simple_text_df['text'])
        
        augmentor = TextAugmentor(augment_factor=3.0, random_state=42)
        result = augmentor.fit_transform(container)
        
        new_texts = set(result.data['text'])
        
        # Должны появиться новые тексты
        new_unique = new_texts - original_texts
        assert len(new_unique) > 0
    

class TestTextAugmentorBalancing:
    """Тесты балансировки классов"""
    
    def test_balance_classes(self, imbalanced_text_df):
        """Тест балансировки классов"""
        container = make_text_container(imbalanced_text_df)
        original_distribution = imbalanced_text_df['label'].value_counts().to_dict()
        
        augmentor = TextAugmentor(balance_classes=True, random_state=42)
        result = augmentor.fit_transform(container)
        
        new_distribution = result.data['label'].value_counts().to_dict()
        
        # Минорный класс должен увеличиться
        assert new_distribution.get(1, 0) >= original_distribution[1]
        
        # Классы должны стать более сбалансированными
        original_ratio = min(original_distribution.values()) / max(original_distribution.values())
        new_ratio = min(new_distribution.values()) / max(new_distribution.values())
        
        assert new_ratio >= original_ratio
        
        print(f"\n✅ Балансировка:")
        print(f"   До: {original_distribution}")
        print(f"   После: {new_distribution}")
    
    def test_balance_multiclass(self, multiclass_text_df):
        """Тест балансировки мультиклассового датасета"""
        container = make_text_container(multiclass_text_df, target_col='category')
        
        augmentor = TextAugmentor(balance_classes=True, random_state=42)
        result = augmentor.fit_transform(container)
        
        # Все классы должны сохраниться
        original_classes = set(multiclass_text_df['category'].unique())
        new_classes = set(result.data['category'].unique())
        
        assert original_classes == new_classes
    
    def test_balance_without_target(self, simple_text_df):
        """Тест балансировки без target колонки"""
        df = simple_text_df.drop(columns=['label'])
        container = DataContainer(data=df.copy(), text_column='text')
        
        augmentor = TextAugmentor(balance_classes=True, random_state=42)
        
        # Не должно падать
        result = augmentor.fit_transform(container)
        assert len(result.data) > 0


class TestTextAugmentorMethods:
    """Тесты методов аугментации"""
    
    def test_eda_method(self, simple_text_df):
        """Тест EDA метода"""
        container = make_text_container(simple_text_df)
        
        config = TextConfig(augment_methods=["eda"])
        augmentor = TextAugmentor(
            config=config,
            augment_factor=2.0,
            random_state=42
        )
        
        result = augmentor.fit_transform(container)
        
        assert "eda" in augmentor._available_methods
        assert len(result.data) > len(simple_text_df)
    
    def test_synonym_wordnet_method(self, simple_text_df):
        """Тест WordNet synonym метода"""
        pytest.importorskip("nltk")
        
        container = make_text_container(simple_text_df)
        
        config = TextConfig(augment_methods=["synonym_wordnet"])
        augmentor = TextAugmentor(
            config=config,
            augment_factor=2.0,
            random_state=42
        )
        
        result = augmentor.fit_transform(container)
        
        if "synonym_wordnet" in augmentor._available_methods:
            assert len(result.data) > len(simple_text_df)
    
    def test_pronoun_to_noun_method(self):
        """Тест pronoun to noun метода"""
        df = pd.DataFrame({
            'text': [
                'He went to the store yesterday.',
                'She said it was important.',
                'They are going home now.',
            ],
            'label': [0, 1, 0]
        })
        
        container = make_text_container(df)
        
        config = TextConfig(augment_methods=["pronoun_to_noun"])
        augmentor = TextAugmentor(
            config=config,
            augment_factor=2.0,
            random_state=42
        )
        
        result = augmentor.fit_transform(container)
        
        assert "pronoun_to_noun" in augmentor._available_methods
        assert len(result.data) > len(df)


class TestTextAugmentorSingleText:
    """Тесты аугментации одиночного текста"""
    
    def test_augment_single(self):
        """Тест augment_single"""
        augmentor = TextAugmentor(random_state=42)
        augmentor._init_methods()
        
        text = "The quick brown fox jumps over the lazy dog."
        result = augmentor.augment_single(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_augment_single_with_method(self):
        """Тест augment_single с указанием метода"""
        augmentor = TextAugmentor(random_state=42)
        augmentor._init_methods()
        
        text = "He went to the store."
        result = augmentor.augment_single(text, method="pronoun_to_noun")
        
        assert isinstance(result, str)
    
    def test_augment_batch(self):
        """Тест augment_batch"""
        augmentor = TextAugmentor(random_state=42)
        
        texts = [
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence.",
        ]
        
        results = augmentor.augment_batch(texts, n_augmentations=2)
        
        assert len(results) == len(texts) * 2


class TestTextAugmentorEdgeCases:
    """Тесты граничных случаев"""
    
    def test_short_text(self):
        """Тест с коротким текстом"""
        df = pd.DataFrame({
            'text': ['Hi', 'OK', 'Yes'],
            'label': [0, 1, 0]
        })
        
        container = make_text_container(df)
        
        augmentor = TextAugmentor(augment_factor=2.0, random_state=42)
        result = augmentor.fit_transform(container)
        
        # Не должно падать
        assert len(result.data) >= len(df)
    
    def test_empty_text_column(self):
        """Тест без text колонки"""
        df = pd.DataFrame({
            'other': ['a', 'b', 'c'],
            'label': [0, 1, 0]
        })
        
        container = DataContainer(data=df.copy(), target_column='label')
        
        augmentor = TextAugmentor(random_state=42)
        result = augmentor.fit_transform(container)
        
        # Должно вернуть без изменений
        assert len(result.data) == len(df)
    
    def test_reproducibility(self, simple_text_df):
        """Тест воспроизводимости"""
        container1 = make_text_container(simple_text_df)
        container2 = make_text_container(simple_text_df)
        
        augmentor1 = TextAugmentor(augment_factor=2.0, random_state=42)
        augmentor2 = TextAugmentor(augment_factor=2.0, random_state=42)
        
        result1 = augmentor1.fit_transform(container1)
        result2 = augmentor2.fit_transform(container2)
        
        # Размеры должны совпадать
        assert len(result1.data) == len(result2.data)


# ==================== INTEGRATION TESTS ====================

class TestTextProcessingIntegration:
    """Интеграционные тесты"""
    
    def test_preprocessor_then_augmentor(self, dirty_text_df):
        """Тест последовательной обработки"""
        container = make_text_container(dirty_text_df)
        
        # Сначала очистка
        preprocessor = TextPreprocessor(preprocessing_level="minimal")
        cleaned = preprocessor.fit_transform(container)
        
        # Затем аугментация
        augmentor = TextAugmentor(augment_factor=2.0, random_state=42)
        result = augmentor.fit_transform(cleaned)
        
        # Результат должен быть больше исходного
        assert len(result.data) > len(dirty_text_df)
        
        # HTML должен быть удалён
        for text in result.data['text']:
            assert '<' not in text
    
    def test_full_pipeline_with_balancing(self, imbalanced_text_df):
        """Полный пайплайн с балансировкой"""
        container = make_text_container(imbalanced_text_df)
        
        # Очистка
        preprocessor = TextPreprocessor(preprocessing_level="full")
        cleaned = preprocessor.fit_transform(container)
        
        # Балансировка через аугментацию
        augmentor = TextAugmentor(balance_classes=True, random_state=42)
        result = augmentor.fit_transform(cleaned)
        
        # Классы должны быть сбалансированы
        distribution = result.data['label'].value_counts()
        ratio = distribution.min() / distribution.max()
        
        assert ratio > 0.5  # Минимум 50% баланс
    
    def test_recommendations_added(self, simple_text_df):
        """Тест добавления рекомендаций"""
        container = make_text_container(simple_text_df)
        
        preprocessor = TextPreprocessor()
        result = preprocessor.fit_transform(container)
        
        assert len(result.recommendations) > 0
        
        rec = result.recommendations[-1]
        assert rec.get('type') == 'preprocessing'