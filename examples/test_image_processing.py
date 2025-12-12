# test_image_processing_enhanced.py
"""
Улучшенное тестирование AutoForge на датасете изображений.

Использует:
1. Только библиотеку automl_data для всех операций
2. Минимальные сторонние зависимости
3. CNN на PyTorch для сравнения
4. Автоматическое создание отчётов
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Для серверов без GUI
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Импорт библиотеки
from automl_data import AutoForge, DataContainer
from automl_data.core.config import ImageConfig

print("=" * 70)
print("📷 ТЕСТИРОВАНИЕ ОБРАБОТКИ ИЗОБРАЖЕНИЙ (УЛУЧШЕННАЯ ВЕРСИЯ)")
print("=" * 70)

# ============================================
# 1. СОЗДАНИЕ ТЕСТОВОГО ДАТАСЕТА (СОБСТВЕННЫЙ КОД)
# ============================================

def create_test_dataset_simple(output_dir="test_dataset"):
    """
    Создаёт минимальный тестовый датасет с использованием только automl_data.
    Использует .jpg формат для совместимости с OpenCV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    data = []
    
    print("\n📁 Создание тестового датасета...")
    
    # Создаём сильный дисбаланс для тестирования балансировки
    # Класс 0: 80% данных, Класс 1: 20% данных
    n_class0 = 80  # 80% (уменьшили для быстрого теста)
    n_class1 = 20  # 20%
    
    # Класс 0: Круги
    for i in range(n_class0):
        label = "circle"
        class_id = 0
        
        # Создаём простое изображение через numpy
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Создаём круг через индексацию
        y, x = np.ogrid[:64, :64]
        center_y, center_x = 32, 32
        radius = 20
        
        # Маска круга
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Зелёный круг (BGR формат)
        img[mask] = [0, 200, 0]  # BGR -> зелёный
        
        # Добавляем шум
        noise = np.random.randint(-20, 20, (64, 64, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Сохраняем как JPG
        img_path = output_dir / f"class{class_id}_{i:03d}.jpg"
        
        # Используем matplotlib для сохранения (не требует OpenCV)
        plt.imsave(str(img_path), img.astype(np.uint8))
        
        data.append({
            "image_id": f"circle_{i:03d}",
            "image_path": str(img_path.name),
            "label": label,
            "class_id": class_id,
            "dataset": "train"
        })
    
    # Класс 1: Квадраты
    for i in range(n_class1):
        label = "square"
        class_id = 1
        
        # Создаём простое изображение через numpy
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Создаём квадрат
        x1, y1 = 10, 10
        x2, y2 = 54, 54
        
        # Красный квадрат (BGR формат)
        img[y1:y2, x1:x2] = [0, 0, 200]  # BGR -> красный
        
        # Добавляем шум
        noise = np.random.randint(-20, 20, (64, 64, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Сохраняем как JPG
        img_path = output_dir / f"class{class_id}_{i:03d}.jpg"
        plt.imsave(str(img_path), img.astype(np.uint8))
        
        data.append({
            "image_id": f"square_{i:03d}",
            "image_path": str(img_path.name),
            "label": label,
            "class_id": class_id,
            "dataset": "train"
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "metadata.csv", index=False)
    
    print(f"✅ Создано {len(df)} изображений")
    print(f"   • Класс 0 (круги): {n_class0} ({n_class0/len(df)*100:.1f}%)")
    print(f"   • Класс 1 (квадраты): {n_class1} ({n_class1/len(df)*100:.1f}%)")
    print(f"   • Дисбаланс: {min(n_class0, n_class1)/max(n_class0, n_class1):.2%}")
    print(f"   • Сохранено в: {output_dir}")
    
    return df, output_dir

# ============================================
# 2. ВИЗУАЛИЗАЦИЯ (МИНИМАЛЬНАЯ)
# ============================================

def visualize_minimal(df, output_dir):
    """Минимальная визуализация с использованием automl_data"""
    print("\n👀 МИНИМАЛЬНАЯ ВИЗУАЛИЗАЦИЯ")
    print("-" * 40)
    
    # Используем DataContainer для анализа
    container = DataContainer(
        data=df.copy(),
        target_column="class_id",
        image_column="image_path",
        image_dir=output_dir
    )
    
    # Анализ через библиотеку
    print("\n📊 АНАЛИЗ ДАННЫХ ЧЕРЕЗ DATACONTAINER:")
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
    
    print(f"\n🎯 ЦЕЛЕВАЯ ПЕРЕМЕННАЯ:")
    print(f"   • Target column: {container.target_column}")
    print(f"   • y shape: {container.y.shape if container.y is not None else 'N/A'}")
    
    return container

# ============================================
# 3. ТЕСТИРОВАНИЕ AUTOFORGE (ПОЛНЫЙ ЦИКЛ)
# ============================================

def test_full_cycle(df, output_dir):
    """Полный цикл тестирования AutoForge"""
    print("\n" + "=" * 60)
    print("⚙️ ПОЛНЫЙ ЦИКЛ ТЕСТИРОВАНИЯ AUTOFORGE")
    print("=" * 60)
    
    # Шаг 1: Конфигурация
    print("\n1️⃣ ШАГ: КОНФИГУРАЦИЯ")
    print("-" * 30)
    
    image_config = ImageConfig(
        # Препроцессинг
        target_size=(64, 64),
        normalize=True,
        keep_aspect_ratio=False,  # Проще для тестирования
        
        # Аугментация (всегда включаем для теста)
        augment=True,
        augment_factor=2.0,  # Удвоить датасет
        
        # Методы аугментации
        horizontal_flip=True,
        rotation_range=15,
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        zoom_range=(0.9, 1.1),
        
        # Балансировка
        balance_classes=True
    )
    
    print(f"   • ImageConfig создан")
    print(f"   • augment: {image_config.augment}")
    print(f"   • augment_factor: {image_config.augment_factor}")
    print(f"   • balance_classes: {image_config.balance_classes}")
    
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
        balance=True,
        balance_threshold=0.5,  # Порог для балансировки
        
        # Разбиение
        test_size=0.2,
        stratify=True,
        random_state=42,
        
        # Логирование
        verbose=True
    )
    
    print(f"   • AutoForge создан")
    print(f"   • target: {forge.config.target}")
    print(f"   • task: {forge.config.task.value}")
    
    # Шаг 3: Fit
    print("\n3️⃣ ШАГ: FIT (АНАЛИЗ ДАННЫХ)")
    print("-" * 30)
    
    try:
        forge.fit(df)
        print(f"   • Pipeline построен")
        print(f"   • Шагов в pipeline: {len(forge._pipeline) if forge._pipeline else 0}")
        print(f"   • Data type: {forge._data_type.name}")
    except Exception as e:
        print(f"   ❌ Ошибка при fit: {e}")
        return None, None, None
    
    # Шаг 4: Transform
    print("\n4️⃣ ШАГ: TRANSFORM (ОБРАБОТКА)")
    print("-" * 30)
    
    try:
        result = forge.transform(df)
        print(f"   • Обработка завершена")
        print(f"   • Время: {result.execution_time:.2f} сек")
        print(f"   • Шагов выполнено: {len(result.steps)}")
    except Exception as e:
        print(f"   ❌ Ошибка при transform: {e}")
        return None, None, None
    
    # Шаг 5: Анализ результатов
    print("\n5️⃣ ШАГ: АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("-" * 30)
    
    print(f"\n📊 РЕЗУЛЬТАТЫ ОБРАБОТКИ:")
    print(f"   • Исходный размер: {len(df)}")
    print(f"   • Финальный размер: {len(result.data)}")
    print(f"   • Увеличение: {len(result.data)/len(df):.2f}x")
    print(f"   • Качество данных: {result.quality_score:.1%}")
    
    # Проверка балансировки
    if result.container.class_distribution:
        print(f"\n📈 БАЛАНСИРОВКА КЛАССОВ:")
        counts = list(result.container.class_distribution.values())
        
        for class_name, count in result.container.class_distribution.items():
            percentage = count / len(result.data) * 100
            print(f"   • Класс {class_name}: {count} ({percentage:.1f}%)")
        
        if len(counts) >= 2:
            ratio = min(counts) / max(counts)
            print(f"   • Соотношение: {ratio:.2f}")
            
            if ratio > 0.7:
                print(f"   • ✅ Отличная балансировка")
            elif ratio > 0.5:
                print(f"   • ⚠️  Средняя балансировка")
            else:
                print(f"   • ❌ Плохая балансировка")
    
    # ДЕТАЛЬНАЯ ПРОВЕРКА АУГМЕНТАЦИИ
    print(f"\n🔍 ДЕТАЛЬНАЯ ПРОВЕРКА АУГМЕНТАЦИИ:")
    print(f"   • Колонки в результате: {result.data.columns.tolist()}")
    
    if "_augmented" in result.data.columns:
        aug_count = result.data["_augmented"].sum()
        print(f"   • Колонка '_augmented' найдена!")
        print(f"   • True значений: {aug_count}")
        print(f"   • False значений: {len(result.data) - aug_count}")
        print(f"   • Процент аугментации: {aug_count/len(result.data)*100:.1f}%")
        
        if aug_count > 0:
            print(f"   • ✅ АУГМЕНТАЦИЯ РАБОТАЕТ!")
            # Показываем примеры аугментированных строк
            aug_samples = result.data[result.data["_augmented"]].head(2)
            print(f"   • Примеры аугментированных строк:")
            for idx, row in aug_samples.iterrows():
                print(f"      Строка {idx}: class_id={row.get('class_id', 'N/A')}, label={row.get('label', 'N/A')}")
        else:
            print(f"   • ❌ Аугментация не добавила данные (aug_count=0)")
    else:
        print(f"   • ❌ Колонка '_augmented' не создана")
        print(f"   • ImageAugmentor не добавляет метки аугментации")
        
        # Проверяем есть ли другие признаки аугментации
        new_columns = set(result.data.columns) - set(df.columns)
        if new_columns:
            print(f"   • Новые колонки: {new_columns}")
        else:
            print(f"   • Нет новых колонок - аугментация не произошла")
    
    # Шаг 6: Разбиение на train/test
    print("\n6️⃣ ШАГ: РАЗБИЕНИЕ НА TRAIN/TEST")
    print("-" * 30)
    
    try:
        X_train, X_test, y_train, y_test = result.get_splits(
            test_size=0.2,
            random_state=42,
            stratify=True
        )
        
        print(f"   • X_train: {X_train.shape}")
        print(f"   • X_test: {X_test.shape}")
        print(f"   • y_train: {y_train.shape if y_train is not None else 'N/A'}")
        print(f"   • y_test: {y_test.shape if y_test is not None else 'N/A'}")
        
        # Проверяем распределение классов в сплитах
        if y_train is not None:
            train_counts = y_train.value_counts()
            test_counts = y_test.value_counts() if y_test is not None else pd.Series()
            
            print(f"\n📊 РАСПРЕДЕЛЕНИЕ В SPLITS:")
            print(f"   • Train: {dict(train_counts)}")
            print(f"   • Test: {dict(test_counts)}")
        
        splits_info = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
    except Exception as e:
        print(f"   ⚠️ Ошибка при разбиении: {e}")
        splits_info = None
    
    return result, forge, splits_info

# ============================================
# 4. УЛУЧШЕННАЯ CNN НА PYTORCH
# ============================================

def test_cnn_enhanced(df_raw, result, splits_info, output_dir):  # ДОБАВЬ output_dir здесь
    """Улучшенное тестирование CNN с реальными данными"""
    print("\n" + "=" * 60)
    print("🧠 УЛУЧШЕННОЕ ТЕСТИРОВАНИЕ CNN")
    print("=" * 60)
    
    try:
        # Пытаемся импортировать PyTorch
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader
            from torchvision import transforms
            import cv2
            TORCH_AVAILABLE = True
        except ImportError:
            print("⚠️  PyTorch/torchvision не установлен. Пропускаем CNN тестирование.")
            print("   Установите: pip install torch torchvision opencv-python")
            return None
        
        print("A. ПОДГОТОВКА ДАННЫХ ДЛЯ CNN")
        print("-" * 30)
        
        # Класс для загрузки изображений
        class ImageDataset(Dataset):
            def __init__(self, df, image_dir, transform=None):
                self.df = df
                self.image_dir = Path(image_dir)
                self.transform = transform
            
            def __len__(self):
                return len(self.df)
            
            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                img_path = self.image_dir / row["image_path"]
                
                # Загружаем изображение
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        # Создаём пустое изображение если не загрузилось
                        img = np.zeros((64, 64, 3), dtype=np.uint8)
                    
                    # Ресайз если нужно
                    if img.shape[:2] != (64, 64):
                        img = cv2.resize(img, (64, 64))
                    
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
                    img = torch.zeros((3, 64, 64))
                    label = 0
                    return img, label
        
        # Простая CNN архитектура
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 8 * 8, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        # Функция обучения
        def train_one_epoch(model, dataloader, criterion, optimizer, device):
            model.train()
            running_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            return running_loss / len(dataloader)
        
        # Функция валидации
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
        
        print("\nB. ТЕСТ НА СЫРЫХ ДАННЫХ")
        print("-" * 30)
        
        # Разбиваем сырые данные
        from sklearn.model_selection import train_test_split
        
        train_raw, test_raw = train_test_split(
            df_raw, test_size=0.2, random_state=42, stratify=df_raw['class_id']
        )
        
        # Создаём датасеты
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
        ])
        
        train_dataset_raw = ImageDataset(train_raw, output_dir, transform=transform)
        test_dataset_raw = ImageDataset(test_raw, output_dir, transform=None)
        
        train_loader_raw = DataLoader(train_dataset_raw, batch_size=16, shuffle=True)
        test_loader_raw = DataLoader(test_dataset_raw, batch_size=16, shuffle=False)
        
        # Создаём и обучаем модель
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_raw = SimpleCNN(num_classes=2).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_raw.parameters(), lr=0.001)
        
        print(f"   • Устройство: {device}")
        print(f"   • Параметры модели: {sum(p.numel() for p in model_raw.parameters()):,}")
        
        # Быстрое обучение (2 эпохи)
        raw_train_losses = []
        for epoch in range(2):
            train_loss = train_one_epoch(model_raw, train_loader_raw, criterion, optimizer, device)
            val_loss, val_acc = validate(model_raw, test_loader_raw, criterion, device)
            raw_train_losses.append(train_loss)
            
            print(f"   Epoch {epoch+1}/2: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        raw_final_loss = raw_train_losses[-1]
        
        print("\nC. ТЕСТ НА ОБРАБОТАННЫХ ДАННЫХ")
        print("-" * 30)
        
        if splits_info and result is not None:
            # Используем обработанные данные
            X_train_proc = splits_info['X_train']
            X_test_proc = splits_info['X_test']
            y_train_proc = splits_info['y_train']
            y_test_proc = splits_info['y_test']
            
            # Создаём DataFrame из сплитов
            train_proc = pd.concat([X_train_proc, y_train_proc], axis=1)
            test_proc = pd.concat([X_test_proc, y_test_proc], axis=1)
            
            # Создаём датасеты
            train_dataset_proc = ImageDataset(train_proc, output_dir, transform=transform)
            test_dataset_proc = ImageDataset(test_proc, output_dir, transform=None)
            
            train_loader_proc = DataLoader(train_dataset_proc, batch_size=16, shuffle=True)
            test_loader_proc = DataLoader(test_dataset_proc, batch_size=16, shuffle=False)
            
            # Создаём и обучаем модель
            model_proc = SimpleCNN(num_classes=2).to(device)
            optimizer_proc = optim.Adam(model_proc.parameters(), lr=0.001)
            
            proc_train_losses = []
            for epoch in range(2):
                train_loss = train_one_epoch(model_proc, train_loader_proc, criterion, optimizer_proc, device)
                val_loss, val_acc = validate(model_proc, test_loader_proc, criterion, device)
                proc_train_losses.append(train_loss)
                
                print(f"   Epoch {epoch+1}/2: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            proc_final_loss = proc_train_losses[-1]
        else:
            print("   ⚠️ Нет обработанных данных для тестирования")
            proc_final_loss = raw_final_loss
        
        # Сравнение
        print("\n📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
        print("-" * 30)
        
        if proc_final_loss > 0:
            improvement = ((raw_final_loss - proc_final_loss) / raw_final_loss * 100)
        else:
            improvement = 0
        
        print(f"   • Сырые данные - финальный loss: {raw_final_loss:.4f}")
        print(f"   • Обработанные - финальный loss: {proc_final_loss:.4f}")
        print(f"   • Улучшение: {improvement:+.1f}%")
        
        if improvement > 5:
            print(f"   ✅ Обработка значительно улучшила обучение")
        elif improvement > 0:
            print(f"   ⚠️  Незначительное улучшение")
        else:
            print(f"   ❌ Обработка не улучшила обучение")
        
        return {
            'raw_final_loss': raw_final_loss,
            'proc_final_loss': proc_final_loss,
            'improvement': improvement,
            'device': str(device)
        }
        
    except Exception as e:
        print(f"\n❌ Ошибка в CNN тестировании: {e}")
        import traceback
        traceback.print_exc()
        return None



# ============================================
# 5. СОЗДАНИЕ ОТЧЁТА
# ============================================

def create_automl_report(df_raw, result, cnn_results=None):
    """Создание отчёта"""
    print("\n" + "=" * 60)
    print("📄 СОЗДАНИЕ ОТЧЁТА")
    print("=" * 60)
    
    report_lines = []
    
    report_lines.append("=" * 70)
    report_lines.append("ОТЧЁТ ПО ТЕСТИРОВАНИЮ AUTOFORGE")
    report_lines.append("=" * 70)
    report_lines.append(f"Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    if result is None:
        report_lines.append("❌ Тестирование не выполнено из-за ошибок")
        report_text = "\n".join(report_lines)
        
        with open("automl_forge_test_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        
        print("⚠️ Отчёт содержит только информацию об ошибках")
        return report_text
    
    # 1. Общая информация
    report_lines.append("1. ОБЩАЯ ИНФОРМАЦИЯ")
    report_lines.append("-" * 40)
    report_lines.append(f"• Тип задачи: {result.config.task.value}")
    report_lines.append(f"• Целевая переменная: {result.config.target}")
    report_lines.append(f"• Исходный размер: {len(df_raw)}")
    report_lines.append(f"• Финальный размер: {len(result.data)}")
    report_lines.append(f"• Увеличение: {len(result.data)/len(df_raw):.2f}x")
    report_lines.append(f"• Время обработки: {result.execution_time:.2f} сек")
    report_lines.append(f"• Качество данных: {result.quality_score:.1%}")
    
    # 2. Шаги обработки
    report_lines.append("\n2. ШАГИ ОБРАБОТКИ")
    report_lines.append("-" * 40)
    for i, step in enumerate(result.steps, 1):
        report_lines.append(f"{i}. {step}")
    
    # 3. Проверка аугментации
    report_lines.append("\n3. ПРОВЕРКА АУГМЕНТАЦИИ")
    report_lines.append("-" * 40)
    
    if "_augmented" in result.data.columns:
        aug_count = result.data["_augmented"].sum()
        if aug_count > 0:
            report_lines.append(f"✅ Аугментация успешно применена")
            report_lines.append(f"   • Аугментированных изображений: {aug_count}")
            report_lines.append(f"   • Оригинальных изображений: {len(result.data) - aug_count}")
            report_lines.append(f"   • Процент аугментации: {aug_count/len(result.data)*100:.1f}%")
        else:
            report_lines.append("⚠️  Аугментация настроена, но не применена")
    else:
        report_lines.append("❌ Аугментация не обнаружена в результатах")
    
    # 4. CNN результаты
    if cnn_results:
        report_lines.append("\n4. РЕЗУЛЬТАТЫ CNN ТЕСТИРОВАНИЯ")
        report_lines.append("-" * 40)
        report_lines.append(f"• Устройство: {cnn_results.get('device', 'N/A')}")
        report_lines.append(f"• Сырые данные - финальный loss: {cnn_results['raw_final_loss']:.4f}")
        report_lines.append(f"• Обработанные - финальный loss: {cnn_results['proc_final_loss']:.4f}")
        report_lines.append(f"• Улучшение: {cnn_results['improvement']:+.1f}%")
        
        if cnn_results['improvement'] > 5:
            report_lines.append("• ВЫВОД: Обработка значительно улучшила обучение CNN")
        elif cnn_results['improvement'] > 0:
            report_lines.append("• ВЫВОД: Незначительное улучшение обучения CNN")
        else:
            report_lines.append("• ВЫВОД: Обработка не улучшила обучение CNN")
    
    # 5. Выводы
    report_lines.append("\n5. ВЫВОДЫ")
    report_lines.append("-" * 40)
    
    # Проверяем работу аугментации
    if "_augmented" in result.data.columns and result.data["_augmented"].sum() > 0:
        report_lines.append("✅ АУГМЕНТАЦИЯ РАБОТАЕТ КОРРЕКТНО")
        report_lines.append(f"   • Библиотека успешно увеличила датасет в {len(result.data)/len(df_raw):.2f} раза")
    else:
        report_lines.append("❌ ПРОБЛЕМА С АУГМЕНТАЦИЕЙ")
        report_lines.append("   • ImageAugmentor не добавил новые изображения")
    
    # Качество данных
    if result.quality_score > 0.8:
        report_lines.append("✅ Высокое качество данных")
    elif result.quality_score > 0.6:
        report_lines.append("⚠️  Среднее качество данных")
    else:
        report_lines.append("❌ Низкое качество данных")
    report_lines.append(f"   • Оценка качества: {result.quality_score:.1%}")
    
    # Сохраняем отчёт
    report_text = "\n".join(report_lines)
    
    with open("automl_forge_test_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"✅ Отчёт сохранён: automl_forge_test_report.txt")
    
    # Выводим краткую версию
    print("\n📋 КРАТКИЙ ОТЧЁТ:")
    print("-" * 40)
    for line in report_lines[:20]:  # Первые 20 строк
        print(f"   {line}")
    
    return report_text

# ============================================
# 6. ГЛАВНАЯ ФУНКЦИЯ ТЕСТИРОВАНИЯ
# ============================================

# ============================================
# 6. ГЛАВНАЯ ФУНКЦИЯ ТЕСТИРОВАНИЯ
# ============================================

def main():
    """Главная функция тестирования"""
    print("\n" + "=" * 70)
    print("🚀 ЗАПУСК УЛУЧШЕННОГО ТЕСТИРОВАНИЯ AUTOFORGE")
    print("=" * 70)
    
    try:
        # 1. Создание датасета
        print("\n1️⃣ ЭТАП: СОЗДАНИЕ МИНИМАЛЬНОГО ДАТАСЕТА")
        df_raw, output_dir = create_test_dataset_simple("automl_test_dataset")
        
        # 2. Минимальная визуализация
        print("\n2️⃣ ЭТАП: МИНИМАЛЬНЫЙ АНАЛИЗ")
        container = visualize_minimal(df_raw, output_dir)
        
        # 3. Полный цикл AutoForge
        print("\n3️⃣ ЭТАП: ПОЛНЫЙ ЦИКЛ AUTOFORGE")
        result, forge, splits_info = test_full_cycle(df_raw, output_dir)
        
        # 4. CNN тестирование (ПЕРЕДАЕМ output_dir)
        print("\n4️⃣ ЭТАП: CNN ТЕСТИРОВАНИЕ")
        cnn_results = test_cnn_enhanced(df_raw, result, splits_info, output_dir)  # Добавь output_dir здесь
        
        # 5. Создание отчёта
        print("\n5️⃣ ЭТАП: СОЗДАНИЕ ОТЧЁТА")
        report = create_automl_report(df_raw, result, cnn_results)
        
        # 6. Сохранение HTML отчёта
        print("\n6️⃣ ЭТАП: HTML ОТЧЁТ AUTOFORGE")
        result.save_report("automl_forge_full_report.html")
        print("✅ HTML отчёт сохранён: automl_forge_full_report.html")
        
        # 7. Итоги
        print("\n" + "=" * 70)
        print("🎉 УЛУЧШЕННОЕ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
        print("=" * 70)
        
        print("\n📊 КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ:")
        print(f"   • Обработано изображений: {len(df_raw)} → {len(result.data)}")
        print(f"   • Увеличение датасета: {len(result.data)/len(df_raw):.2f}x")
        print(f"   • Время обработки: {result.execution_time:.2f} сек")
        print(f"   • Качество данных: {result.quality_score:.1%}")
        print(f"   • Шагов обработки: {len(result.steps)}")
        
        if cnn_results:
            print(f"   • CNN улучшение: {cnn_results['improvement']:+.1f}%")
        
        print("\n📁 СОЗДАННЫЕ ФАЙЛЫ:")
        print("   1. automl_test_dataset/ - Тестовый датасет")
        print("   2. automl_forge_test_report.txt - Текстовый отчёт")
        print("   3. automl_forge_full_report.html - HTML отчёт")
        
        print("\n✅ Все тесты выполнены с использованием библиотеки automl_data!")
        
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
    # Минимальная конфигурация matplotlib
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
    
    # Запуск тестирования
    success = main()
    
    if success:
        print("\n✅ Тестирование успешно завершено!")
        print("   Библиотека automl_data работает корректно.")
    else:
        print("\n❌ Тестирование завершилось с ошибками.")
    
    print("\n" + "=" * 70)