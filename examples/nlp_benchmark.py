# examples/debug_augmentation_fixed.py
"""
Исправленный отладочный скрипт для проверки аугментации.
"""

import pandas as pd
import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_data.core.forge import AutoForge

print("=" * 80)
print("🛠️  ИСПРАВЛЕННЫЙ ТЕСТ АУГМЕНТАЦИИ")
print("=" * 80)

# Тестовые данные с дисбалансом
df = pd.DataFrame({
    'text': [
        'Это хороший продукт очень качественный',
        'Отличное качество рекомендую всем',
        'Не понравилось плохое качество',
        'Ужасный товар не советую покупать',
        'Хороший товар за свои деньги',
        'Плохое качество разочарован',
        'Отличный продукт буду покупать еще',
        'Ужасное качество деньги на ветер',
        'Нормально но есть недостатки',
        'Прекрасный товар всем рекомендую',
        'Средний товар ничего особенного',
        'Хороший но дорогой'
    ],
    'sentiment': ['positive', 'positive', 'negative', 'negative', 
                  'positive', 'negative', 'positive', 'negative',
                  'neutral', 'positive', 'neutral', 'positive']
})

from automl_data import AutoForge

# Для трансформеров (BERT, RoBERTa) - минимальная предобработка
forge = AutoForge(
    target="sentiment",
    text_column="text",
    text_preprocessing_level="minimal",  # Только очистка HTML, URL, пробелов
    text_augment=True,
    text_augment_factor=2.0,
    verbose=True
)
result = forge.fit_transform(df)

# Для классических методов (TF-IDF, Word2Vec) - полная предобработка
forge = AutoForge(
    target="sentiment",
    text_column="text",
    text_preprocessing_level="full",  # + lowercase, stopwords, lemmatize
    text_augment=True,
    text_balance_classes=True,  # Балансировка классов
    text_augment_methods=["eda", "synonym_wordnet", "t5_paraphrase"],
    verbose=True
)
result = forge.fit_transform(df)