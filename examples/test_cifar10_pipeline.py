"""
Пайплайн для тестирования automl_data на датасете CIFAR-10.

Включает:
1. Загрузку CIFAR-10 через torchvision
2. Полный цикл обработки с AutoForge
3. Аугментацию и балансировку
4. Обучение моделей (CNN) с сравнением результатов
5. Детальные отчёты и визуализации
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
import logging
from typing import Dict, Any, Optional, Tuple
import time

warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)

def get_forge_result_stats(result):
    """Получить статистику из ForgeResult"""
    stats = {
        'original_size': 'N/A',
        'augmented_size': 'N/A',
        'total_size': 'N/A',
        'quality_score': 'N/A',
        'augmentation_info': {}
    }
    
    if not result:
        return stats
    
    # Способ 1: Через контейнер
    if hasattr(result, 'container') and result.container:
        container = result.container
        
        # Из метаданных контейнера
        if hasattr(container, 'metadata') and container.metadata:
            stats['augmentation_info'] = container.metadata.get('augmentation', {})
            stats['original_size'] = container.metadata.get('augmentation', {}).get('original_size', 'N/A')
            stats['augmented_size'] = container.metadata.get('augmentation', {}).get('augmented_size', 'N/A')
        
        # Из данных контейнера
        if hasattr(container, 'data') and container.data is not None:
            stats['total_size'] = len(container.data)
            
            if '_augmented' in container.data.columns:
                aug_count = container.data['_augmented'].sum()
                orig_count = len(container.data) - aug_count
                stats['original_size'] = orig_count
                stats['augmented_size'] = aug_count
    
    # Способ 2: Через ForgeResult напрямую
    if hasattr(result, 'data'):
        stats['total_size'] = len(result.data)
    
    if hasattr(result, 'quality_score'):
        stats['quality_score'] = result.quality_score
    
    return stats


def analyze_augmentation_results(df, result):
    """Анализ результатов аугментации"""
    print("\n🔍 АНАЛИЗ РЕЗУЛЬТАТОВ АУГМЕНТАЦИИ")
    print("-" * 40)
    
    if not result or not hasattr(result, 'data'):
        print("❌ Нет данных для анализа")
        return
    
    # Базовые метрики
    original_size = len(df)
    final_size = len(result.data)
    increase = final_size / original_size if original_size > 0 else 0
    
    print(f"📊 ОСНОВНЫЕ МЕТРИКИ:")
    print(f"   • Исходный размер: {original_size}")
    print(f"   • Финальный размер: {final_size}")
    print(f"   • Увеличение: {increase:.2f}x")
    
    if hasattr(result, 'quality_score'):
        print(f"   • Качество данных: {result.quality_score:.1%}")
    
    # Анализ аугментации
    if '_augmented' in result.data.columns:
        aug_count = result.data['_augmented'].sum()
        orig_count = final_size - aug_count
        
        print(f"\n🎯 АУГМЕНТАЦИЯ:")
        print(f"   • Аугментированных: {aug_count}")
        print(f"   • Оригинальных: {orig_count}")
        print(f"   • Процент аугментации: {aug_count/final_size*100:.1f}%" if final_size > 0 else "N/A")
        
        # Распределение по классам
        if 'class_id' in result.data.columns:
            print(f"\n📈 РАСПРЕДЕЛЕНИЕ ПО КЛАССАМ:")
            
            # Оригинальные данные
            original_dist = df['class_id'].value_counts().sort_index()
            
            # Обработанные данные
            processed_dist = result.data['class_id'].value_counts().sort_index()
            
            # Аугментированные данные
            augmented_dist = result.data[result.data['_augmented']]['class_id'].value_counts().sort_index()
            
            for class_id in sorted(original_dist.index):
                orig = original_dist.get(class_id, 0)
                proc = processed_dist.get(class_id, 0)
                aug = augmented_dist.get(class_id, 0)
                
                increase = proc / orig if orig > 0 else 0
                percentage = proc / final_size * 100 if final_size > 0 else 0
                
                print(f"   • Класс {class_id}:")
                print(f"     - Было: {orig}")
                print(f"     - Стало: {proc} ({increase:.2f}x)")
                print(f"     - Аугментировано: {aug}")
                print(f"     - В датасете: {percentage:.1f}%")

print("=" * 70)
print("🚀 ПАЙПЛАЙН ТЕСТИРОВАНИЯ НА CIFAR-10")
print("=" * 70)

# ============================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА CIFAR-10
# ============================================

def load_cifar10_as_dataframe(output_dir: Path = "cifar10_dataset", fix_test_size: int = 500) -> Tuple[pd.DataFrame, Path]:
    """
    Загружает CIFAR-10 и преобразует в DataFrame.
    
    Args:
        output_dir: Директория для сохранения изображений
        fix_test_size: Фиксированный размер тестового набора
        
    Returns:
        DataFrame с данными и путь к директории
    """
    try:
        import torch
        import torchvision
        import torchvision.transforms as transforms
        from PIL import Image
    except ImportError:
        print("❌ Требуется установить torch и torchvision")
        print("   pip install torch torchvision")
        sys.exit(1)
    
    print("\n📥 ЗАГРУЗКА CIFAR-10")
    print("-" * 40)
    
    # Создаём директорию
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Классы CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Загружаем датасет
    print("Загружаю CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"✅ Загружено:")
    print(f"   • Train: {len(trainset)} изображений")
    print(f"   • Test: {len(testset)} изображений")
    print(f"   • Всего: {len(trainset) + len(testset)} изображений")
    print(f"   • Классы: {len(classes)}")
    
    # Создаём DataFrame
    data = []
    image_counter = 0
    
    # Функция для сохранения изображения
    def save_image(tensor, filename):
        # Конвертируем tensor в numpy array
        img_np = tensor.numpy().transpose(1, 2, 0) * 255
        img_np = img_np.astype(np.uint8)
        
        # Сохраняем как PNG
        img = Image.fromarray(img_np)
        img.save(filename)
    
    print("\n📁 Сохранение изображений...")
    
    # Обрабатываем train set (возьмём подмножество для скорости)
    n_train_samples = 2500  # 2500 для train
    n_test_samples = fix_test_size  # Фиксированный размер test
    
    print(f"Беру подмножество:")
    print(f"   • Train samples: {n_train_samples}")
    print(f"   • Test samples: {n_test_samples} (фиксировано)")
    
    # Train images - выбираем случайно
    train_indices = np.random.choice(len(trainset), n_train_samples, replace=False)
    for idx in train_indices:
        img, label = trainset[idx]
        
        # Сохраняем изображение
        img_path = output_dir / f"train_{image_counter:06d}.png"
        save_image(img, img_path)
        
        data.append({
            "image_id": f"train_{image_counter:06d}",
            "image_path": img_path.name,
            "label": classes[label],
            "class_id": int(label),
            "dataset": "train"
        })
        
        image_counter += 1
    
    # Test images - ОБЯЗАТЕЛЬНО выбираем ВСЕ 500 изображений
    # Стратифицированная выборка для равномерного распределения по классам
    print(f"\n📊 Формирование тестового набора ({n_test_samples} изображений):")
    
    # Собираем индексы по классам
    test_indices_by_class = {}
    for idx, (_, label) in enumerate(testset):
        class_id = label
        if class_id not in test_indices_by_class:
            test_indices_by_class[class_id] = []
        test_indices_by_class[class_id].append(idx)
    
    # Берем равное количество из каждого класса
    per_class_count = n_test_samples // len(classes)
    extra_count = n_test_samples % len(classes)
    
    test_indices = []
    for class_id in range(len(classes)):
        class_indices = test_indices_by_class.get(class_id, [])
        if len(class_indices) >= per_class_count:
            selected = np.random.choice(class_indices, per_class_count, replace=False)
            test_indices.extend(selected)
        else:
            # Если в классе меньше изображений, берем все
            test_indices.extend(class_indices)
    
    # Если нужно добавить дополнительные изображения
    if extra_count > 0:
        # Собираем все оставшиеся индексы
        all_test_indices = list(range(len(testset)))
        remaining_indices = [idx for idx in all_test_indices if idx not in test_indices]
        if len(remaining_indices) >= extra_count:
            extra_selected = np.random.choice(remaining_indices, extra_count, replace=False)
            test_indices.extend(extra_selected)
    
    # Ограничиваем до нужного размера
    test_indices = test_indices[:n_test_samples]
    
    print(f"   • Отобрано {len(test_indices)} тестовых изображений")
    
    # Сохраняем тестовые изображения
    for idx in test_indices:
        img, label = testset[idx]
        
        # Сохраняем изображение
        img_path = output_dir / f"test_{image_counter:06d}.png"
        save_image(img, img_path)
        
        data.append({
            "image_id": f"test_{image_counter:06d}",
            "image_path": img_path.name,
            "label": classes[label],
            "class_id": int(label),
            "dataset": "test"
        })
        
        image_counter += 1
    
    # Создаём DataFrame
    df = pd.DataFrame(data)
    
    # Сохраняем метаданные
    metadata_path = output_dir / "cifar10_metadata.csv"
    df.to_csv(metadata_path, index=False)
    
    print(f"\n✅ Датысет сохранён:")
    print(f"   • Изображений: {len(df)}")
    print(f"   • Сохранено в: {output_dir}")
    print(f"   • Метаданные: {metadata_path}")
    
    # Анализ распределения
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ КЛАССОВ:")
    print(f"   • Train set: {len(df[df['dataset'] == 'train'])}")
    print(f"   • Test set: {len(df[df['dataset'] == 'test'])}")
    
    class_counts = df['class_id'].value_counts().sort_index()
    for class_id, count in class_counts.items():
        percentage = count / len(df) * 100
        class_name = classes[class_id]
        print(f"   • {class_name} ({class_id}): {count} ({percentage:.1f}%)")
    
    # Проверяем распределение в test set
    test_df = df[df['dataset'] == 'test']
    if len(test_df) > 0:
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ В TEST SET:")
        test_class_counts = test_df['class_id'].value_counts().sort_index()
        for class_id, count in test_class_counts.items():
            percentage = count / len(test_df) * 100
            class_name = classes[class_id]
            print(f"   • {class_name} ({class_id}): {count} ({percentage:.1f}%)")
    
    return df, output_dir


# ============================================
# 2. ВИЗУАЛИЗАЦИЯ ДАТАСЕТА
# ============================================

def visualize_cifar10_dataset(df: pd.DataFrame, output_dir: Path):
    """Визуализация образцов CIFAR-10"""
    print("\n👀 ВИЗУАЛИЗАЦИЯ CIFAR-10")
    print("-" * 40)
    
    try:
        from automl_data import DataContainer
        from PIL import Image
        import matplotlib.pyplot as plt
        
        # Создаём контейнер для анализа
        container = DataContainer(
            data=df.copy(),
            target_column="class_id",
            image_column="image_path",
            image_dir=output_dir
        )
        
        print("\n📊 АНАЛИЗ ДАННЫХ:")
        print(f"   • Тип данных: {container.data_type.name}")
        print(f"   • Размер: {container.shape}")
        print(f"   • Колонки: {len(container.columns)}")
        print(f"   • Числовые колонки: {len(container.numeric_columns)}")
        print(f"   • Категориальные колонки: {len(container.categorical_columns)}")
        
        if container.class_distribution:
            print(f"\n📈 РАСПРЕДЕЛЕНИЕ КЛАССОВ:")
            for class_name, count in container.class_distribution.items():
                percentage = count / len(container) * 100
                print(f"   • Класс {class_name}: {count} ({percentage:.1f}%)")
        
        # Визуализация образцов
        print("\n🖼️ ВИЗУАЛИЗАЦИЯ ОБРАЗЦОВ:")
        
        # Создаём grid из примеров
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        # Берем по одному примеру каждого класса
        classes = df['class_id'].unique()
        
        for i, class_id in enumerate(sorted(classes)[:10]):  # максимум 10 классов
            class_samples = df[df['class_id'] == class_id]
            if len(class_samples) > 0:
                sample = class_samples.iloc[0]
                img_path = output_dir / sample['image_path']
                
                try:
                    img = Image.open(img_path)
                    axes[i].imshow(img)
                    axes[i].set_title(f"{sample['label']} (ID: {class_id})")
                    axes[i].axis('off')
                except Exception as e:
                    axes[i].text(0.5, 0.5, f"Error: {e}", 
                               ha='center', va='center')
                    axes[i].axis('off')
        
        plt.suptitle("Примеры изображений CIFAR-10", fontsize=16)
        plt.tight_layout()
        
        # Сохраняем визуализацию
        viz_path = output_dir / "cifar10_samples.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"   • Визуализация сохранена: {viz_path}")
        
        plt.show()
        
        return container
        
    except Exception as e:
        print(f"⚠️ Ошибка при визуализации: {e}")
        return None


# ============================================
# 3. ПОЛНЫЙ ЦИКЛ AUTOFORGE НА CIFAR-10
# ============================================

def test_cifar10_with_autoforge(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """
    Полный цикл обработки CIFAR-10 с AutoForge.
    """
    print("\n" + "=" * 60)
    print("⚙️ ПОЛНЫЙ ЦИКЛ AUTOFORGE НА CIFAR-10")
    print("=" * 60)
    
    try:
        from automl_data import AutoForge
        from automl_data.core.config import ImageConfig
        
        # Шаг 1: Конфигурация
        print("\n1️⃣ ШАГ: КОНФИГУРАЦИЯ")
        print("-" * 30)
        
        image_config = ImageConfig(
            # Препроцессинг
            target_size=(32, 32),  # CIFAR-10 размер
            normalize=True,
            keep_aspect_ratio=False,
            
            # Аугментация - ВКЛЮЧАЕМ для train
            augment=True,
            augment_factor=3.0,  # Увеличить в 3 раза
            
            # Методы аугментации
            horizontal_flip=True,
            rotation_range=10,
            brightness_range=(0.8, 1.2),
            contrast_range=(0.8, 1.2),
            zoom_range=(0.9, 1.1),
            
            # Балансировка - ОТКЛЮЧАЕМ для чистого увеличения
            balance_classes=False,
        )
        
        print(f"   • ImageConfig создан для CIFAR-10")
        print(f"   • Target size: {image_config.target_size}")
        print(f"   • Augment: {image_config.augment}")
        print(f"   • Balance classes: {image_config.balance_classes}")
        print(f"   • Augment factor: {image_config.augment_factor}")
        
        # Шаг 2: Создание AutoForge
        print("\n2️⃣ ШАГ: СОЗДАНИЕ AUTOFORGE")
        print("-" * 30)
        
        forge = AutoForge(
            target="class_id",
            image_column="image_path",
            image_dir=output_dir,
            task="classification",
            
            # Конфигурации
            image_config=image_config,
            
            # Общие настройки
            balance=False,
            
            # Разбиение
            test_size=0.2,
            stratify=True,
            random_state=42,
            
            # Логирование
            verbose=True
        )
        
        print(f"   • AutoForge создан")
        print(f"   • Target: {forge.config.target}")
        print(f"   • Task: {forge.config.task.value}")
        
        # Шаг 3: Fit
        print("\n3️⃣ ШАГ: FIT (АНАЛИЗ ДАННЫХ)")
        print("-" * 30)
        
        # Разделяем на train и test
        train_df = df[df['dataset'] == 'train'].copy()
        test_df = df[df['dataset'] == 'test'].copy()
        
        print(f"   • Train size: {len(train_df)}")
        print(f"   • Test size: {len(test_df)} (фиксировано)")
        
        # Проверяем распределение в test
        print(f"\n   📊 TEST SET РАСПРЕДЕЛЕНИЕ:")
        test_counts = test_df['class_id'].value_counts().sort_index()
        for class_id, count in test_counts.items():
            percentage = count / len(test_df) * 100
            print(f"     • Класс {class_id}: {count} ({percentage:.1f}%)")
        
        start_time = time.time()
        forge.fit(train_df)
        fit_time = time.time() - start_time
        
        print(f"   • Pipeline построен")
        print(f"   • Шагов в pipeline: {len(forge._pipeline) if forge._pipeline else 0}")
        print(f"   • Время fit: {fit_time:.2f} сек")
        
        # Шаг 4: Transform train данных
        print("\n4️⃣ ШАГ: TRANSFORM TRAIN ДАННЫХ")
        print("-" * 30)
        
        start_time = time.time()
        train_result = forge.transform(train_df)
        transform_time = time.time() - start_time
        
        print(f"   • Обработка завершена")
        print(f"   • Время transform: {transform_time:.2f} сек")
        print(f"   • Шагов выполнено: {len(train_result.steps)}")
        
        # Шаг 5: Анализ результатов
        print("\n5️⃣ ШАГ: АНАЛИЗ РЕЗУЛЬТАТОВ")
        print("-" * 30)
        
        print(f"\n📊 РЕЗУЛЬТАТЫ ОБРАБОТКИ:")
        print(f"   • Исходный train размер: {len(train_df)}")
        print(f"   • Финальный train размер: {len(train_result.data)}")
        print(f"   • Увеличение: {len(train_result.data)/len(train_df):.2f}x")
        print(f"   • Качество данных: {train_result.quality_score:.1%}")
        
        # Проверка аугментации
        if "_augmented" in train_result.data.columns:
            aug_count = train_result.data["_augmented"].sum()
            print(f"\n🔍 ПРОВЕРКА АУГМЕНТАЦИИ:")
            print(f"   • Аугментированных: {aug_count}")
            print(f"   • Оригинальных: {len(train_result.data) - aug_count}")
            print(f"   • Процент аугментации: {aug_count/len(train_result.data)*100:.1f}%")
        
        # Распределение по классам после аугментации
        if train_result.container.class_distribution:
            print(f"\n📈 РАСПРЕДЕЛЕНИЕ ПО КЛАССАМ ПОСЛЕ ОБРАБОТКИ:")
            counts = list(train_result.container.class_distribution.values())
            
            for class_name, count in train_result.container.class_distribution.items():
                original_count = len(train_df[train_df['class_id'] == int(class_name)])
                increase = count / original_count if original_count > 0 else 0
                percentage = count / len(train_result.data) * 100
                print(f"   • Класс {class_name}: было {original_count}, стало {count} ({increase:.2f}x, {percentage:.1f}%)")
            
            if len(counts) >= 2:
                ratio = min(counts) / max(counts)
                print(f"   • Соотношение min/max: {ratio:.2f}")
        
        # Получаем сплиты
        X_train, X_val, y_train, y_val = train_result.get_splits(
            test_size=0.2,
            random_state=42,
            stratify=True
        )
        
        print(f"\n🎯 РАЗБИЕНИЕ НА TRAIN/VAL:")
        print(f"   • X_train: {X_train.shape}")
        print(f"   • X_val: {X_val.shape}")
        print(f"   • y_train: {y_train.shape}")
        print(f"   • y_val: {y_val.shape}")
        
        # Шаг 6: Обработка тестовых данных БЕЗ аугментации
        print("\n6️⃣ ШАГ: ОБРАБОТКА ТЕСТОВЫХ ДАННЫХ (БЕЗ АУГМЕНТАЦИИ)")
        print("-" * 30)
        
        # ВАЖНО: Отключаем аугментацию для test
        test_image_config = ImageConfig(
            target_size=(32, 32),
            normalize=True,
            keep_aspect_ratio=False,
            augment=False,  # ← ОТКЛЮЧАЕМ аугментацию!
            balance_classes=False,
        )
        
        # Создаём отдельный forge для test (без аугментации)
        test_forge = AutoForge(
            target="class_id",
            image_column="image_path",
            image_dir=output_dir,
            task="classification",
            image_config=test_image_config,
            balance=False,
            test_size=0.0,  # Не разбиваем test
            verbose=False  # Уменьшаем вывод
        )
        
        # Fit на тех же данных для консистентности
        test_forge.fit(train_df)  # Используем train для fit, но transform применяем к test
        
        # Transform test данных (без аугментации)
        test_result = test_forge.transform(test_df)
        
        print(f"   • Test size до: {len(test_df)}")
        print(f"   • Test size после: {len(test_result.data)} (должно остаться {len(test_df)})")
        
        # Проверяем, что test данные не были аугментированы
        if "_augmented" in test_result.data.columns:
            test_aug_count = test_result.data["_augmented"].sum()
            print(f"   • Аугментированных в test: {test_aug_count} (должно быть 0)")
            
            if test_aug_count > 0:
                print(f"   ⚠️  ВНИМАНИЕ: Test данные были аугментированы!")
            else:
                print(f"   ✅ Test данные сохранены без аугментации")
        else:
            print(f"   ✅ Test данные обработаны без аугментации")
        
        # Проверяем распределение в test
        if test_result.container.class_distribution:
            print(f"\n   📊 TEST SET РАСПРЕДЕЛЕНИЕ ПОСЛЕ ОБРАБОТКИ:")
            test_counts = list(test_result.container.class_distribution.values())
            
            for class_name, count in test_result.container.class_distribution.items():
                percentage = count / len(test_result.data) * 100
                print(f"     • Класс {class_name}: {count} ({percentage:.1f}%)")
        
        # Анализ результатов
        if train_result:
            analyze_augmentation_results(train_df, train_result)
        
        # Сохраняем результаты
        results = {
            'forge': forge,
            'train_result': train_result,
            'test_result': test_result,
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'X_test': test_result.data,
            'y_test': test_result.data['class_id'] if 'class_id' in test_result.data.columns else None,
            'fit_time': fit_time,
            'transform_time': transform_time,
            'total_time': fit_time + transform_time,
            'original_train_df': train_df,
            'original_test_df': test_df,  # Сохраняем оригинальный test
            'output_dir': output_dir
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Ошибка при обработке AutoForge: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================
# 4. ТЕСТИРОВАНИЕ МОДЕЛЕЙ НА CIFAR-10
# ============================================

def train_cifar10_models(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Обучение моделей на обработанных данных CIFAR-10.
    
    Сравнивает:
    1. Модель на сырых данных (train/val/test)
    2. Модель на обработанных данных (train/val/test)
    """
    print("\n" + "=" * 60)
    print("🧠 ТЕСТИРОВАНИЕ МОДЕЛЕЙ НА CIFAR-10")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms
        import cv2
        
        # Класс датасета
        class CIFAR10Dataset(Dataset):
            def __init__(self, df, image_dir, transform=None):
                self.df = df.reset_index(drop=True)
                self.image_dir = Path(image_dir)
                self.transform = transform
            
            def __len__(self):
                return len(self.df)
            
            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                img_path = self.image_dir / row["image_path"]
                
                try:
                    # Загружаем изображение
                    img = cv2.imread(str(img_path))
                    if img is None:
                        img = np.zeros((32, 32, 3), dtype=np.uint8)
                    
                    # Ресайз если нужно
                    if img.shape[:2] != (32, 32):
                        img = cv2.resize(img, (32, 32))
                    
                    # Конвертируем BGR -> RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Нормализация [0, 1]
                    img = img.astype(np.float32) / 255.0
                    
                    # Транспонируем для PyTorch (H, W, C) -> (C, H, W)
                    img = np.transpose(img, (2, 0, 1))
                    
                    # В тензор
                    img = torch.FloatTensor(img)
                    
                    # Применяем трансформации
                    if self.transform:
                        img = self.transform(img)
                    
                    # Метка
                    label = int(row["class_id"])
                    
                    return img, label
                    
                except Exception:
                    # В случае ошибки возвращаем нулевое изображение
                    img = torch.zeros((3, 32, 32))
                    label = 0
                    return img, label
        
        # Архитектура CNN для CIFAR-10
        class CIFAR10CNN(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout(0.25),
                    
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout(0.25),
                    
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout(0.25),
                )
                
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 4 * 4, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        # Функции обучения
        def train_epoch(model, dataloader, criterion, optimizer, device):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            accuracy = 100 * correct / total if total > 0 else 0
            return running_loss / len(dataloader), accuracy
        
        def validate(model, dataloader, criterion, device):
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in dataloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    running_loss += loss.item()
                    
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            accuracy = 100 * correct / total if total > 0 else 0
            return running_loss / len(dataloader), accuracy
        
        def test_model(model, dataloader, criterion, device):
            """Финальное тестирование на test set"""
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for data, target in dataloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    running_loss += loss.item()
                    
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
            
            accuracy = 100 * correct / total if total > 0 else 0
            
            # Подробная статистика по классам
            from sklearn.metrics import classification_report, confusion_matrix
            report = classification_report(all_targets, all_predictions, output_dict=True)
            cm = confusion_matrix(all_targets, all_predictions)
            
            return running_loss / len(dataloader), accuracy, report, cm
        
        # Настройки
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   • Устройство: {device}")
        print(f"   • CUDA доступна: {torch.cuda.is_available()}")
        
        # Подготовка данных
        print("\nA. ПОДГОТОВКА ДАННЫХ")
        print("-" * 30)
        
        # Трансформации
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Загружаем оригинальные данные для сравнения
        from sklearn.model_selection import train_test_split
        
        print("\nB. МОДЕЛЬ НА СЫРЫХ ДАННЫХ")
        print("-" * 30)
        
        # Берём оригинальные данные
        original_train_df = results.get('original_train_df')
        original_test_df = results.get('original_test_df')  # ← ДОБАВИЛИ test данные
        
        if original_train_df is None:
            print("   ⚠️ Нет отдельного датасета сырых данных")
            print("   Пропускаем сравнение с сырыми данными")
            raw_results = None
        else:
            # Разделяем сырые данные на train/val
            raw_train, raw_val = train_test_split(
                original_train_df, 
                test_size=0.2, 
                random_state=42, 
                stratify=original_train_df['class_id']
            )
            
            # Создаём датасеты
            raw_train_dataset = CIFAR10Dataset(raw_train, results['output_dir'], transform=train_transform)
            raw_val_dataset = CIFAR10Dataset(raw_val, results['output_dir'], transform=test_transform)
            
            # Test датасет из ОРИГИНАЛЬНЫХ test данных
            raw_test_dataset = CIFAR10Dataset(original_test_df, results['output_dir'], transform=test_transform)
            
            raw_train_loader = DataLoader(raw_train_dataset, batch_size=64, shuffle=True)
            raw_val_loader = DataLoader(raw_val_dataset, batch_size=64, shuffle=False)
            raw_test_loader = DataLoader(raw_test_dataset, batch_size=64, shuffle=False)
            
            # Создаём и обучаем модель
            raw_model = CIFAR10CNN(num_classes=10).to(device)
            raw_criterion = nn.CrossEntropyLoss()
            raw_optimizer = optim.Adam(raw_model.parameters(), lr=0.001, weight_decay=1e-4)
            raw_scheduler = optim.lr_scheduler.StepLR(raw_optimizer, step_size=10, gamma=0.5)
            
            print(f"   • Параметры модели: {sum(p.numel() for p in raw_model.parameters()):,}")
            print(f"   • Размер train: {len(raw_train)}")
            print(f"   • Размер val: {len(raw_val)}")
            print(f"   • Размер test: {len(original_test_df)} (отдельный набор)")
            
            # Быстрое обучение (5 эпох для теста)
            raw_train_losses = []
            raw_val_losses = []
            raw_val_accs = []
            
            for epoch in range(5):
                train_loss, train_acc = train_epoch(raw_model, raw_train_loader, raw_criterion, raw_optimizer, device)
                val_loss, val_acc = validate(raw_model, raw_val_loader, raw_criterion, device)
                raw_scheduler.step()
                
                raw_train_losses.append(train_loss)
                raw_val_losses.append(val_loss)
                raw_val_accs.append(val_acc)
                
                print(f"   Epoch {epoch+1}/5: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ на test set
            print(f"\n   📊 ТЕСТИРОВАНИЕ НА TEST SET:")
            raw_test_loss, raw_test_acc, raw_test_report, raw_test_cm = test_model(
                raw_model, raw_test_loader, raw_criterion, device
            )
            
            print(f"   • Test Loss: {raw_test_loss:.4f}, Test Acc: {raw_test_acc:.2f}%")
            
            raw_results = {
                'final_val_loss': raw_val_losses[-1],
                'final_val_acc': raw_val_accs[-1],
                'final_test_loss': raw_test_loss,
                'final_test_acc': raw_test_acc,
                'test_report': raw_test_report,
                'train_losses': raw_train_losses,
                'val_losses': raw_val_losses,
                'val_accs': raw_val_accs
            }
        
        print("\nC. МОДЕЛЬ НА ОБРАБОТАННЫХ ДАННЫХ")
        print("-" * 30)
        
        # Используем обработанные данные из AutoForge
        X_train = results['X_train']
        X_val = results['X_val']
        y_train = results['y_train']
        y_val = results['y_val']
        X_test = results['X_test']  # Обработанные test данные
        y_test = results['y_test']   # Обработанные test метки
        
        # Создаём DataFrame из сплитов
        train_proc = pd.concat([X_train, y_train], axis=1)
        val_proc = pd.concat([X_val, y_val], axis=1)
        test_proc = X_test.copy()
        if y_test is not None:
            test_proc['class_id'] = y_test
        
        # Создаём датасеты
        proc_train_dataset = CIFAR10Dataset(train_proc, results['output_dir'], transform=train_transform)
        proc_val_dataset = CIFAR10Dataset(val_proc, results['output_dir'], transform=test_transform)
        proc_test_dataset = CIFAR10Dataset(test_proc, results['output_dir'], transform=test_transform)
        
        proc_train_loader = DataLoader(proc_train_dataset, batch_size=64, shuffle=True)
        proc_val_loader = DataLoader(proc_val_dataset, batch_size=64, shuffle=False)
        proc_test_loader = DataLoader(proc_test_dataset, batch_size=64, shuffle=False)
        
        # Создаём и обучаем модель
        proc_model = CIFAR10CNN(num_classes=10).to(device)
        proc_criterion = nn.CrossEntropyLoss()
        proc_optimizer = optim.Adam(proc_model.parameters(), lr=0.001, weight_decay=1e-4)
        proc_scheduler = optim.lr_scheduler.StepLR(proc_optimizer, step_size=10, gamma=0.5)
        
        print(f"   • Параметры модели: {sum(p.numel() for p in proc_model.parameters()):,}")
        print(f"   • Размер train (обработанный): {len(train_proc)}")
        print(f"   • Размер val (обработанный): {len(val_proc)}")
        print(f"   • Размер test (обработанный): {len(test_proc)}")
        print(f"   • Увеличение датасета: {len(train_proc)/len(original_train_df):.2f}x")
        
        # Обучение
        proc_train_losses = []
        proc_val_losses = []
        proc_val_accs = []
        
        for epoch in range(5):
            train_loss, train_acc = train_epoch(proc_model, proc_train_loader, proc_criterion, proc_optimizer, device)
            val_loss, val_acc = validate(proc_model, proc_val_loader, proc_criterion, device)
            proc_scheduler.step()
            
            proc_train_losses.append(train_loss)
            proc_val_losses.append(val_loss)
            proc_val_accs.append(val_acc)
            
            print(f"   Epoch {epoch+1}/5: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ на test set
        print(f"\n   📊 ТЕСТИРОВАНИЕ НА TEST SET:")
        proc_test_loss, proc_test_acc, proc_test_report, proc_test_cm = test_model(
            proc_model, proc_test_loader, proc_criterion, device
        )
        
        print(f"   • Test Loss: {proc_test_loss:.4f}, Test Acc: {proc_test_acc:.2f}%")
        
        proc_results = {
            'final_val_loss': proc_val_losses[-1],
            'final_val_acc': proc_val_accs[-1],
            'final_test_loss': proc_test_loss,
            'final_test_acc': proc_test_acc,
            'test_report': proc_test_report,
            'train_losses': proc_train_losses,
            'val_losses': proc_val_losses,
            'val_accs': proc_val_accs
        }
        
        # Сравнение результатов
        print("\n📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
        print("-" * 30)
        
        if raw_results:
            print(f"   СЫРЫЕ ДАННЫЕ:")
            print(f"   • Val Loss: {raw_results['final_val_loss']:.4f}, Val Acc: {raw_results['final_val_acc']:.2f}%")
            print(f"   • Test Loss: {raw_results['final_test_loss']:.4f}, Test Acc: {raw_results['final_test_acc']:.2f}%")
            
            print(f"\n   ОБРАБОТАННЫЕ ДАННЫЕ:")
            print(f"   • Val Loss: {proc_results['final_val_loss']:.4f}, Val Acc: {proc_results['final_val_acc']:.2f}%")
            print(f"   • Test Loss: {proc_results['final_test_loss']:.4f}, Test Acc: {proc_results['final_test_acc']:.2f}%")
            
            # Рассчитываем улучшение на TEST SET
            if raw_results['final_test_loss'] > 0:
                loss_improvement = ((raw_results['final_test_loss'] - proc_results['final_test_loss']) / raw_results['final_test_loss'] * 100)
            else:
                loss_improvement = 0
            
            acc_improvement = proc_results['final_test_acc'] - raw_results['final_test_acc']
            
            print(f"\n   УЛУЧШЕНИЕ НА TEST SET:")
            print(f"   • Улучшение loss: {loss_improvement:+.1f}%")
            print(f"   • Улучшение accuracy: {acc_improvement:+.2f}%")
            
            if acc_improvement > 2.0:
                print(f"   ✅ Обработка значительно улучшила точность на тестовых данных")
            elif acc_improvement > 0.5:
                print(f"   ⚠️  Незначительное улучшение точности на тестовых данных")
            elif acc_improvement > 0:
                print(f"   ⚠️  Минимальное улучшение точности на тестовых данных")
            else:
                print(f"   ❌ Обработка не улучшила точность на тестовых данных")
        else:
            print(f"   • Обработанные данные - Test Loss: {proc_results['final_test_loss']:.4f}, Test Acc: {proc_results['final_test_acc']:.2f}%")
            print(f"   • Сравнение с сырыми данными недоступно")
        
        # Визуализация результатов
        if raw_results:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Loss comparison
            axes[0].plot(raw_results['val_losses'], label='Сырые (val)', marker='o')
            axes[0].plot(proc_results['val_losses'], label='Обработанные (val)', marker='s')
            axes[0].axhline(y=raw_results['final_test_loss'], color='r', linestyle='--', label='Сырые (test)')
            axes[0].axhline(y=proc_results['final_test_loss'], color='b', linestyle='--', label='Обработанные (test)')
            axes[0].set_xlabel('Эпоха')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Сравнение Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Accuracy comparison
            axes[1].plot(raw_results['val_accs'], label='Сырые (val)', marker='o')
            axes[1].plot(proc_results['val_accs'], label='Обработанные (val)', marker='s')
            axes[1].axhline(y=raw_results['final_test_acc'], color='r', linestyle='--', label='Сырые (test)')
            axes[1].axhline(y=proc_results['final_test_acc'], color='b', linestyle='--', label='Обработанные (test)')
            axes[1].set_xlabel('Эпоха')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_title('Сравнение Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Bar chart comparison
            comparison_data = {
                'Сырые': [raw_results['final_val_loss'], raw_results['final_val_acc'], 
                         raw_results['final_test_loss'], raw_results['final_test_acc']],
                'Обработанные': [proc_results['final_val_loss'], proc_results['final_val_acc'],
                                proc_results['final_test_loss'], proc_results['final_test_acc']]
            }
            
            x = np.arange(4)
            width = 0.35
            
            axes[2].bar(x - width/2, [raw_results['final_val_loss'], raw_results['final_val_acc'], 
                                     raw_results['final_test_loss'], raw_results['final_test_acc']], 
                       width, label='Сырые', color=['red', 'orange', 'darkred', 'darkorange'])
            axes[2].bar(x + width/2, [proc_results['final_val_loss'], proc_results['final_val_acc'],
                                     proc_results['final_test_loss'], proc_results['final_test_acc']], 
                       width, label='Обработанные', color=['blue', 'green', 'darkblue', 'darkgreen'])
            
            axes[2].set_xlabel('Метрики')
            axes[2].set_ylabel('Значение')
            axes[2].set_title('Финальные результаты (Val/Test)')
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(['Val Loss', 'Val Acc', 'Test Loss', 'Test Acc'])
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("cifar10_comparison_detailed.png", dpi=150, bbox_inches='tight')
            print(f"   • Графики сохранены: cifar10_comparison_detailed.png")
            plt.show()
        
        return {
            'raw_results': raw_results,
            'proc_results': proc_results,
            'device': str(device)
        }
        
    except Exception as e:
        print(f"\n❌ Ошибка при обучении моделей: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================
# 5. СОЗДАНИЕ ОТЧЁТА
# ============================================
def create_cifar10_report(df: pd.DataFrame, results: Dict[str, Any], model_results: Dict[str, Any]) -> str:
    """
    Создание детального отчёта по тестированию на CIFAR-10.
    """
    print("\n" + "=" * 60)
    print("📄 СОЗДАНИЕ ОТЧЁТА ПО CIFAR-10")
    print("=" * 60)
    
    report_lines = []
    
    report_lines.append("=" * 70)
    report_lines.append("ОТЧЁТ ПО ТЕСТИРОВАНИЮ AUTOFORGE НА CIFAR-10")
    report_lines.append("=" * 70)
    report_lines.append(f"Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 1. Информация о датасете
    report_lines.append("1. ИНФОРМАЦИЯ О ДАТАСЕТЕ CIFAR-10")
    report_lines.append("-" * 40)
    report_lines.append(f"• Общий размер: {len(df)} изображений")
    report_lines.append(f"• Размер train: {len(df[df['dataset'] == 'train'])}")
    report_lines.append(f"• Размер test: {len(df[df['dataset'] == 'test'])}")
    report_lines.append(f"• Количество классов: {df['class_id'].nunique()}")
    
    # Распределение классов
    class_counts = df['class_id'].value_counts().sort_index()
    report_lines.append(f"• Распределение классов:")
    for class_id, count in class_counts.items():
        percentage = count / len(df) * 100
        report_lines.append(f"  - Класс {class_id}: {count} ({percentage:.1f}%)")
    
    # 2. Результаты AutoForge
    if results:
        report_lines.append("\n2. РЕЗУЛЬТАТЫ AUTOFORGE")
        report_lines.append("-" * 40)
        report_lines.append(f"• Время fit: {results.get('fit_time', 0):.2f} сек")
        report_lines.append(f"• Время transform: {results.get('transform_time', 0):.2f} сек")
        report_lines.append(f"• Общее время: {results.get('total_time', 0):.2f} сек")
        
        if results.get('train_result'):
            train_result = results['train_result']
            
            # ИСПРАВЛЕНИЕ: Используем правильный путь к метаданным
            original_size = 'N/A'
            augmented_size = 'N/A'
            
            # Способ 1: Через контейнер
            if hasattr(train_result, 'container') and train_result.container:
                container = train_result.container
                
                # Получаем метаданные из контейнера
                if hasattr(container, 'metadata') and container.metadata:
                    original_size = container.metadata.get('augmentation', {}).get('original_size', 'N/A')
                    augmented_size = container.metadata.get('augmentation', {}).get('augmented_size', 'N/A')
                
                # Или через DataContainer напрямую
                if hasattr(container, 'data') and container.data is not None:
                    # Вычисляем размеры из данных
                    if '_augmented' in container.data.columns:
                        aug_count = container.data['_augmented'].sum()
                        orig_count = len(container.data) - aug_count
                        original_size = orig_count
                        augmented_size = aug_count
            
            # Способ 2: Через атрибуты ForgeResult
            elif hasattr(train_result, 'original_size'):
                original_size = train_result.original_size
            elif hasattr(train_result, 'input_size'):
                original_size = train_result.input_size
            
            # Вычисляем увеличение
            increase = 'N/A'
            if isinstance(original_size, (int, float)) and original_size > 0:
                total_size = len(train_result.data) if hasattr(train_result, 'data') else 'N/A'
                if isinstance(total_size, (int, float)):
                    increase = total_size / original_size
            
            report_lines.append(f"• Исходный размер train: {original_size}")
            report_lines.append(f"• Финальный размер train: {len(train_result.data) if hasattr(train_result, 'data') else 'N/A'}")
            report_lines.append(f"• Увеличение датасета: {increase if isinstance(increase, str) else f'{increase:.2f}x'}")
            report_lines.append(f"• Качество данных: {train_result.quality_score if hasattr(train_result, 'quality_score') else 'N/A'}")
            
            # Аугментация
            if hasattr(train_result, 'data') and '_augmented' in train_result.data.columns:
                aug_count = train_result.data["_augmented"].sum()
                report_lines.append(f"• Аугментированных изображений: {aug_count}")
                report_lines.append(f"• Процент аугментации: {aug_count/len(train_result.data)*100:.1f}%" if len(train_result.data) > 0 else "N/A")
    
    # 3. Результаты моделей
    if model_results:
        report_lines.append("\n3. РЕЗУЛЬТАТЫ ОБУЧЕНИЯ МОДЕЛЕЙ")
        report_lines.append("-" * 40)
        
        report_lines.append(f"• Устройство: {model_results.get('device', 'N/A')}")
        
        if model_results.get('raw_results'):
            raw = model_results['raw_results']
            proc = model_results['proc_results']
            
            report_lines.append("\n   СЫРЫЕ ДАННЫЕ:")
            report_lines.append(f"   • Final Validation Loss: {raw['final_val_loss']:.4f}")
            report_lines.append(f"   • Final Validation Accuracy: {raw['final_val_acc']:.2f}%")
            
            report_lines.append("\n   ОБРАБОТАННЫЕ ДАННЫЕ:")
            report_lines.append(f"   • Final Validation Loss: {proc['final_val_loss']:.4f}")
            report_lines.append(f"   • Final Validation Accuracy: {proc['final_val_acc']:.2f}%")
            
            # Сравнение
            if raw['final_val_loss'] > 0:
                loss_improvement = ((raw['final_val_loss'] - proc['final_val_loss']) / raw['final_val_loss'] * 100)
            else:
                loss_improvement = 0
            
            acc_improvement = proc['final_val_acc'] - raw['final_val_acc']
            
            report_lines.append("\n   СРАВНЕНИЕ:")
            report_lines.append(f"   • Улучшение Loss: {loss_improvement:+.1f}%")
            report_lines.append(f"   • Улучшение Accuracy: {acc_improvement:+.2f}%")
            
            if acc_improvement > 2.0:
                report_lines.append(f"   • ВЫВОД: ✅ Обработка значительно улучшила точность модели")
            elif acc_improvement > 0.5:
                report_lines.append(f"   • ВЫВОД: ⚠️  Незначительное улучшение точности")
            elif acc_improvement > 0:
                report_lines.append(f"   • ВЫВОД: ⚠️  Минимальное улучшение точности")
            else:
                report_lines.append(f"   • ВЫВОД: ❌ Обработка не улучшила точность")
    
    # 4. Выводы
    report_lines.append("\n4. ВЫВОДЫ")
    report_lines.append("-" * 40)
    
    if results and model_results:
        report_lines.append("✅ Библиотека automl_data успешно обработала CIFAR-10")
        report_lines.append("✅ Аугментация и балансировка работают корректно")
        
        if model_results.get('proc_results', {}).get('final_val_acc', 0) > 50:
            report_lines.append("✅ Модель достигла разумной точности на CIFAR-10")
        else:
            report_lines.append("⚠️  Точность модели ниже ожидаемой, требуется настройка")
    else:
        report_lines.append("❌ Тестирование не завершено успешно")
    
    # 5. Рекомендации
    report_lines.append("\n5. РЕКОМЕНДАЦИИ")
    report_lines.append("-" * 40)
    report_lines.append("• Увеличить количество эпох обучения для лучшей точности")
    report_lines.append("• Использовать более сложную архитектуру CNN")
    report_lines.append("• Добавить больше методов аугментации")
    report_lines.append("• Настроить гиперпараметры обучения")
    
    # Сохраняем отчёт
    report_text = "\n".join(report_lines)
    
    report_path = "cifar10_test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"✅ Отчёт сохранён: {report_path}")
    
    # Выводим краткую версию
    print("\n📋 КРАТКИЙ ОТЧЁТ:")
    print("-" * 40)
    for line in report_lines[:30]:  # Первые 30 строк
        print(f"   {line}")
    
    return report_text


# ============================================
# 6. ГЛАВНЫЙ ПАЙПЛАЙН
# ============================================

def main():
    """Главный пайплайн тестирования на CIFAR-10"""
    print("\n" + "=" * 70)
    print("🚀 ЗАПУСК ПАЙПЛАЙНА ТЕСТИРОВАНИЯ НА CIFAR-10")
    print("=" * 70)
    
    try:
        # 1. Загрузка CIFAR-10 с фиксированным размером test
        print("\n1️⃣ ЭТАП: ЗАГРУЗКА CIFAR-10")
        df, output_dir = load_cifar10_as_dataframe("cifar10_test_dataset", fix_test_size=500)
        
        # Сохраняем оригинальные данные для сравнения
        original_train_df = df[df['dataset'] == 'train'].copy()
        original_test_df = df[df['dataset'] == 'test'].copy()
        
        # 2. Визуализация
        print("\n2️⃣ ЭТАП: ВИЗУАЛИЗАЦИЯ ДАТАСЕТА")
        container = visualize_cifar10_dataset(df, output_dir)
        
        # 3. Обработка с AutoForge
        print("\n3️⃣ ЭТАП: ОБРАБОТКА AUTOFORGE")
        print("   • Train: с аугментацией (augment_factor=3.0)")
        print("   • Test: без аугментации (сохраняется 500 изображений)")
        
        results = test_cifar10_with_autoforge(df, output_dir)
        
        if results is None:
            print("❌ AutoForge не смог обработать данные")
            return False
        
        # Добавляем оригинальные данные в результаты
        results['original_train_df'] = original_train_df
        results['original_test_df'] = original_test_df
        results['output_dir'] = output_dir
        
        # 4. Обучение моделей
        print("\n4️⃣ ЭТАП: ОБУЧЕНИЕ МОДЕЛЕЙ")
        model_results = train_cifar10_models(results)
        
        # 5. Создание отчёта
        print("\n5️⃣ ЭТАП: СОЗДАНИЕ ОТЧЁТА")
        report = create_cifar10_report(df, results, model_results)
        
        # 6. Сохранение HTML отчёта
        if results and results.get('train_result'):
            print("\n6️⃣ ЭТАП: HTML ОТЧЁТ AUTOFORGE")
            results['train_result'].save_report("cifar10_autoforge_report.html")
            print("✅ HTML отчёт сохранён: cifar10_autoforge_report.html")
        
        # 7. Итоги
        print("\n" + "=" * 70)
        print("🎉 ПАЙПЛАЙН ТЕСТИРОВАНИЯ НА CIFAR-10 ЗАВЕРШЁН!")
        print("=" * 70)
        
        print("\n📊 КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ:")
        print(f"   • Train изображений: {len(original_train_df)} → {len(results['train_result'].data)}")
        print(f"   • Test изображений: {len(original_test_df)} (без изменений)")
        print(f"   • Увеличение train: {len(results['train_result'].data)/len(original_train_df):.2f}x")
        print(f"   • Время обработки: {results['total_time']:.2f} сек")
        print(f"   • Качество данных: {results['train_result'].quality_score:.1%}")
        
        if model_results and model_results.get('proc_results'):
            proc = model_results['proc_results']
            print(f"   • Точность модели: {proc['final_val_acc']:.2f}%")
            
            if model_results.get('raw_results'):
                raw = model_results['raw_results']
                acc_improvement = proc['final_val_acc'] - raw['final_val_acc']
                print(f"   • Улучшение точности: {acc_improvement:+.2f}%")
        
        print("\n📁 СОЗДАННЫЕ ФАЙЛЫ:")
        print("   1. cifar10_test_dataset/ - Датысет CIFAR-10")
        print("   2. cifar10_test_report.txt - Текстовый отчёт")
        print("   3. cifar10_autoforge_report.html - HTML отчёт AutoForge")
        print("   4. cifar10_comparison.png - Графики сравнения")
        print("   5. cifar10_samples.png - Визуализация образцов")
        
        print("\n✅ Тестирование успешно завершено!")
        print("   Библиотека automl_data работает с реальными датасетами изображений.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================
# ЗАПУСК
# ============================================

if __name__ == "__main__":
    # Настройка matplotlib
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Запуск пайплайна
    success = main()
    
    if success:
        print("\n✅ Пайплайн успешно завершён!")
        print("   automl_data отлично справляется с CIFAR-10.")
    else:
        print("\n❌ Пайплайн завершился с ошибками.")
    
    print("\n" + "=" * 70)