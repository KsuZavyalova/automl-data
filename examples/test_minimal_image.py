# test_image_minimal.py
"""
Минимальный тест automl_data на изображениях.
Только AutoForge, без дополнительных конфигов.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import shutil


def create_test_dataset(n_images: int = 50, output_dir: str = "test_images"):
    """Создаёт синтетический датасет изображений"""
    
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    print(f"📁 Создаю {n_images} тестовых изображений...")
    
    data = []
    classes = ["cat", "dog", "bird"]
    
    for i in range(n_images):
        label = classes[i % len(classes)]
        class_id = i % len(classes)
        
        # Создаём цветное изображение 32x32
        colors = {"cat": [255, 100, 100], "dog": [100, 255, 100], "bird": [100, 100, 255]}
        base_color = colors[label]
        
        img_array = np.random.randint(0, 50, (32, 32, 3), dtype=np.uint8)
        img_array += np.array(base_color, dtype=np.uint8)
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        filename = f"img_{i:04d}.png"
        Image.fromarray(img_array).save(output_dir / filename)
        
        data.append({
            "image_path": filename,
            "label": label,
            "class_id": class_id
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Создано {n_images} изображений")
    print(f"   Классы: {df['label'].value_counts().to_dict()}")
    
    return df, output_dir


# ============================================
# ТЕСТ
# ============================================

print("🚀 МИНИМАЛЬНЫЙ ТЕСТ AUTOFORGE НА ИЗОБРАЖЕНИЯХ")
print("=" * 50)

# 1. Создаём датасет
df, image_dir = create_test_dataset(n_images=100)

print(f"\n📊 Исходные данные: {len(df)} изображений")

# 2. Импортируем и запускаем AutoForge
from automl_data import AutoForge

# Минимальный вызов с аугментацией
forge = AutoForge(
    target="class_id",
    image_column="image_path",
    image_dir=image_dir,
    
    # Параметры аугментации (передаются напрямую)
    augment=True,
    augment_factor=2.0,
    
    verbose=True
)

# 3. Обрабатываем
result = forge.fit_transform(df)

# 4. Результаты
print(f"\n" + "=" * 50)
print("📊 РЕЗУЛЬТАТЫ")
print("=" * 50)
print(f"• Было: {len(df)}")
print(f"• Стало: {len(result.data)}")
print(f"• Увеличение: {len(result.data)/len(df):.2f}x")
print(f"• Quality: {result.quality_score:.1%}")
print(f"• Время: {result.execution_time:.2f}s")

# Проверяем аугментацию
if '_augmented' in result.data.columns:
    aug_count = result.data['_augmented'].sum()
    print(f"\n✅ Аугментация работает!")
    print(f"   • Оригинальных: {len(result.data) - aug_count}")
    print(f"   • Аугментированных: {aug_count}")
else:
    print(f"\n⚠️ Аугментация не сработала")
    print(f"   Колонки: {list(result.data.columns)}")

# Сплиты
X_train, X_val, y_train, y_val = result.get_splits()
print(f"\n🎯 Сплиты: train={len(X_train)}, val={len(X_val)}")

print("\n✅ Тест завершён!")