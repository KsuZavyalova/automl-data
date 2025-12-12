# examples/ruwordnet_demo.py
"""
Демонстрация использования RuWordNet в текстовой аугментации.
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_data.core.forge import AutoForge
from automl_data.core.config import TextConfig

print("=" * 70)
print("ДЕМОНСТРАЦИЯ RuWordNet В АУГМЕНТАЦИИ ТЕКСТА")
print("=" * 70)

# Создаём русскоязычный датасет
df = pd.DataFrame({
    'text': [
        'Хороший качественный продукт с отличными характеристиками',
        'Плохое обслуживание и низкое качество товара',
        'Средний продукт за свои деньги, ничего особенного',
        'Прекрасный товар, полностью оправдал ожидания',
        'Ужасное качество, очень разочарован покупкой'
    ],
    'rating': [5, 1, 3, 5, 1]
})

print(f"Исходные данные: {len(df)} строк")
print(f"Распределение оценок: {df['rating'].value_counts().to_dict()}")

# Создаём конфигурацию с RuWordNet
config = TextConfig(
    preprocessing_level='normal',
    language='ru',
    augment=True,
    augment_factor=2.0,
    augment_methods=['synonym_ruwordnet', 'eda'],  # RuWordNet в приоритете
    use_ruwordnet=True,
    ruwordnet_cache_dir='./ruwordnet_cache',  # Кэш для ускорения
    balance_classes=True
)

# Создаём AutoForge с RuWordNet
forge = AutoForge(
    target='rating',
    text_column='text',
    text_config=config,
    verbose=True
)

print("\nЗапуск аугментации с RuWordNet...")
result = forge.fit_transform(df)

print(f"\nРезультаты:")
print(f"До: {len(df)} строк")
print(f"После: {len(result.data)} строк")
print(f"Увеличение: {len(result.data)/len(df):.1f}x")

# Показываем примеры аугментированных текстов
if len(result.data) > len(df):
    augmented = result.data.iloc[len(df):]
    print(f"\nПримеры аугментированных текстов (RuWordNet):")
    
    for i, (idx, row) in enumerate(augmented.head(3).iterrows(), 1):
        print(f"\nПример {i}:")
        orig_idx = idx - len(df)
        if orig_idx < len(df):
            print(f"  Оригинал: {df.iloc[orig_idx]['text']}")
        print(f"  Аугмент:  {row['text']}")

# Проверяем рекомендации
print("\nРекомендации:")
for rec in result.recommendations:
    if rec.get('type') == 'text_augmentation':
        print(f"- Использован RuWordNet: {rec.get('ruwordnet_used', False)}")
        print(f"- Методы: {rec.get('methods_used', {})}")

print("\n" + "=" * 70)
print("ПРЕИМУЩЕСТВА RuWordNet:")
print("1. Лингвистически корректные синонимы")
print("2. Большой словарный запас")
print("3. Учёт семантических отношений между словами")
print("4. Кэширование для повышения производительности")
print("=" * 70)