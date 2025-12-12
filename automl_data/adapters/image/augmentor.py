# automl_data/adapters/image/augmentor.py
"""
SOTA аугментация изображений с использованием Albumentations.

Включает:
- Базовые трансформации (flip, rotate, crop)
- Цветовые аугментации
- Геометрические искажения
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging  # ← ДОБАВЬ ИМПОРТ LOGGING

from ..base import BaseAdapter
from ...core.container import DataContainer, ProcessingStage
from ...core.config import ImageConfig
from ...utils.dependencies import require_package, optional_import


class ImageAugmentor(BaseAdapter):
    """
    SOTA аугментация изображений через Albumentations.
    
    Автоматически выбирает набор аугментаций в зависимости от задачи.
    Поддерживает балансировку классов через аугментацию.
    
    Example:
        >>> augmentor = ImageAugmentor(
        ...     augment_factor=3.0,
        ...     use_randaugment=True,
        ...     balance_classes=True
        ... )
        >>> result = augmentor.fit_transform(container)
    """
    
    def __init__(
        self,
        config: ImageConfig | None = None,
        augment_factor: float = 3.0,
        balance_classes: bool = True,
        
        # Базовые аугментации
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        rotation_range: int = 15,
        
        # Цветовые
        brightness_range: tuple[float, float] = (0.8, 1.2),
        contrast_range: tuple[float, float] = (0.8, 1.2),
        saturation_range: tuple[float, float] = (0.8, 1.2),
        hue_range: float = 0.1,
        
        # Геометрические
        zoom_range: tuple[float, float] = (0.9, 1.1),
        shift_range: float = 0.1,
        shear_range: float = 0.1,
        
        # Продвинутые
        use_randaugment: bool = True,
        use_mixup: bool = False,
        use_cutmix: bool = False,
        use_cutout: bool = True,
        
        # Шум и блюр
        add_noise: bool = True,
        add_blur: bool = True,
        
        output_dir: Path | None = None,
        random_state: int = 42,
        
        verbose: bool = True,  # ← ДОБАВЬ ПАРАМЕТР VERBOSE В __init__
        save_examples: bool = True,
        examples_dir: Path | None = None,
        n_examples: int = 10,
        **kwargs
    ):
        super().__init__(name="ImageAugmentor", **kwargs)
        
        # ИНИЦИАЛИЗИРУЕМ ЛОГГЕР
        self._logger = logging.getLogger(f"automl_data.ImageAugmentor")
        self.verbose = verbose  # ← СОХРАНЯЕМ VERBOSE
        
        if config:
            self.augment_factor = config.augment_factor
            self.balance_classes = config.balance_classes
            self.horizontal_flip = config.horizontal_flip
            self.vertical_flip = config.vertical_flip
            self.rotation_range = config.rotation_range
            self.brightness_range = config.brightness_range
            self.contrast_range = config.contrast_range
            self.zoom_range = config.zoom_range
            self.use_randaugment = config.use_randaugment
            self.use_mixup = config.use_mixup
            self.use_cutmix = config.use_cutmix
        else:
            self.augment_factor = augment_factor
            self.balance_classes = balance_classes
            self.horizontal_flip = horizontal_flip
            self.vertical_flip = vertical_flip
            self.rotation_range = rotation_range
            self.brightness_range = brightness_range
            self.contrast_range = contrast_range
            self.zoom_range = zoom_range
            self.use_randaugment = use_randaugment
            self.use_mixup = use_mixup
            self.use_cutmix = use_cutmix
        
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.shift_range = shift_range
        self.shear_range = shear_range
        self.use_cutout = use_cutout
        self.add_noise = add_noise
        self.add_blur = add_blur
        
        self.output_dir = Path(output_dir) if output_dir else None
        self.random_state = random_state
        
        self._transform = None
        self._class_counts: dict = {}
        self._target_count: int = 0
        self._logs: List[str] = []  # ← ДЛЯ ХРАНЕНИЯ ЛОГОВ
        self._augmented_info: Dict[int, Dict] = {}  # ← ДЛЯ ОТСЛЕЖИВАНИЯ АУГМЕНТАЦИЙ

        self.save_examples = save_examples
        self.examples_dir = Path(examples_dir) if examples_dir else None
        self.n_examples = n_examples
        self._examples: list[dict] = []
        
        np.random.seed(self.random_state)
    
    def _fit_impl(self, container: DataContainer) -> None:
        require_package("albumentations", "albumentations")
        require_package("cv2", "opencv-python")
        
        import albumentations as A
        
        np.random.seed(self.random_state)
        
        if self.verbose:
            self._log("🔧 Инициализация аугментаций...")
        
        # Собираем pipeline аугментаций
        transforms = []
        
        # Геометрические
        if self.horizontal_flip:
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if self.vertical_flip:
            transforms.append(A.VerticalFlip(p=0.5))
        
        if self.rotation_range > 0:
            transforms.append(A.Rotate(
                limit=self.rotation_range, 
                p=0.5,
                border_mode=0
            ))
        
        if self.shift_range > 0 or self.zoom_range != (1.0, 1.0):
            transforms.append(A.ShiftScaleRotate(
                shift_limit=self.shift_range,
                scale_limit=(self.zoom_range[0] - 1, self.zoom_range[1] - 1),
                rotate_limit=0,
                p=0.5,
                border_mode=0
            ))
        
        if self.shear_range > 0:
            transforms.append(A.Affine(
                shear=(-self.shear_range * 45, self.shear_range * 45),
                p=0.3
            ))
        
        # Цветовые
        transforms.append(A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(self.brightness_range[0] - 1, self.brightness_range[1] - 1),
                contrast_limit=(self.contrast_range[0] - 1, self.contrast_range[1] - 1),
                p=1.0
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=self.hue_range,
                p=1.0
            ),
        ], p=0.5))
        
        # RandAugment-style
        if self.use_randaugment:
            transforms.append(A.OneOf([
                A.Equalize(p=1.0),
                A.Posterize(p=1.0),
                A.Solarize(threshold=128, p=1.0),
                A.Sharpen(p=1.0),
                A.Emboss(p=1.0),
            ], p=0.3))
        
        # Шум и блюр
        if self.add_noise:
            transforms.append(A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(p=1.0),
            ], p=0.2))
        
        if self.add_blur:
            transforms.append(A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2))
        
        # Cutout
        if self.use_cutout:
            transforms.append(A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.3
            ))
        
        self._transform = A.Compose(transforms)
        
        # Подсчёт классов для балансировки
        if self.balance_classes and container.target_column:
            self._class_counts = container.data[container.target_column].value_counts().to_dict()
            self._target_count = max(self._class_counts.values())
            
            if self.verbose:
                self._log(f"📊 Распределение классов для балансировки:")
                for cls, count in self._class_counts.items():
                    self._log(f"   • Класс {cls}: {count} (нужно до {self._target_count})")
        
        self._fit_info = {
            "n_transforms": len(transforms),
            "augment_factor": self.augment_factor,
            "balance_classes": self.balance_classes
        }
        
        if self.verbose:
            self._log(f"✅ Создан pipeline с {len(transforms)} трансформациями")
    
    def _transform_impl(self, container: DataContainer) -> DataContainer:
        if not container.image_column or self._transform is None:
            self._log("⚠️ Пропускаем аугментацию: нет image_column или transform", level="warning")
            return container
        
        if self.verbose:
            self._log("🔧 Начинаю аугментацию изображений...")
            self._log(f"   • balance_classes: {self.balance_classes}")
            self._log(f"   • augment_factor: {self.augment_factor}")
            self._log(f"   • Исходный размер: {len(container.data)}")
        
        import cv2
        
        df = container.data.copy()
        image_col = container.image_column
        target_col = container.target_column
        image_dir = container.image_dir
        
        # Создаём выходную директорию
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            aug_dir = self.output_dir / "augmented"
            aug_dir.mkdir(exist_ok=True)
        else:
            aug_dir = None
        
        augmented_rows = []
        aug_counter = 0
        self._augmented_info.clear()  # ← ОЧИЩАЕМ ИНФОРМАЦИЮ О ПРЕДЫДУЩИХ АУГМЕНТАЦИЯХ
        
        # Определяем сколько аугментаций нужно
        if self.balance_classes and target_col and self._class_counts:
            if self.verbose:
                self._log("📊 Режим: Балансировка классов")
                self._log(f"   • Распределение классов: {self._class_counts}")
                self._log(f"   • Целевой размер класса: {self._target_count}")
            
            # Балансировка через аугментацию
            for label, count in self._class_counts.items():
                class_df = df[df[target_col] == label]
                n_to_generate = self._target_count - count
                
                if n_to_generate > 0:
                    if self.verbose:
                        self._log(f"   • Класс {label}: нужно добавить {n_to_generate} изображений")
                    
                    generated, aug_counter = self._augment_class(
                        class_df, image_col, image_dir, aug_dir,
                        n_to_generate, aug_counter, original_df=df
                    )
                    
                    if generated:
                        augmented_rows.extend(generated)
                        if self.verbose:
                            self._log(f"   • Класс {label}: добавлено {len(generated)} аугментированных")
                    else:
                        if self.verbose:
                            self._log(f"   • Класс {label}: не удалось сгенерировать изображения")
        else:
            # Простая аугментация
            if self.verbose:
                self._log("📊 Режим: Простая аугментация")
            
            n_to_generate = max(0, int(len(df) * (self.augment_factor - 1)))
            if n_to_generate > 0:
                if self.verbose:
                    self._log(f"   • Нужно сгенерировать: {n_to_generate} изображений")
                
                generated, aug_counter = self._augment_class(
                    df, image_col, image_dir, aug_dir,
                    n_to_generate, aug_counter, original_df=df
                )
                
                if generated:
                    augmented_rows.extend(generated)
                    if self.verbose:
                        self._log(f"   • Добавлено: {len(generated)} аугментированных изображений")
        
        # Объединяем ОРИГИНАЛЬНЫЕ и АУГМЕНТИРОВАННЫЕ данные
        if augmented_rows:
            if self.verbose:
                self._log(f"✅ Найдено {len(augmented_rows)} аугментированных изображений")
            
            # Создаём DataFrame из аугментированных строк
            aug_df = pd.DataFrame(augmented_rows)
            
            # Убедимся, что у аугментированных строк есть ВСЕ колонки из оригинального df
            for col in df.columns:
                if col not in aug_df.columns and col != '_augmented':
                    # Копируем значения из исходных строк
                    def get_original_value(row):
                        source_idx = row.get('_source_idx', -1)
                        if 0 <= source_idx < len(df):
                            return df.iloc[source_idx][col]
                        # Если нет source_idx, пытаемся найти по другим признакам
                        for key in ['image_path', 'image_id']:
                            if key in row and key in df.columns:
                                matching = df[df[key] == row[key]]
                                if len(matching) > 0:
                                    return matching.iloc[0][col]
                        return None
                    
                    aug_df[col] = aug_df.apply(get_original_value, axis=1)
            
            # ГАРАНТИРУЕМ НАЛИЧИЕ КОЛОНКИ _augmented:
            # 1. Оригинальные данные - _augmented = False
            df['_augmented'] = False
            
            # 2. Аугментированные данные - _augmented = True
            aug_df['_augmented'] = True
            
            # Добавляем информацию об аугментации
            aug_df['_augmentation_info'] = aug_df.apply(
                lambda row: self._augmented_info.get(len(df) + row.name, {}) 
                if row.name in self._augmented_info else {},
                axis=1
            )
            
            # Объединяем
            combined_df = pd.concat([df, aug_df], ignore_index=True)
            
            # ОБЯЗАТЕЛЬНО присваиваем обратно в container
            container.data = combined_df
            container.stage = ProcessingStage.AUGMENTED
            
            if self.verbose:
                self._log(f"📈 Итоговый размер: {len(container.data)} изображений")
                self._log(f"   • Оригинальных: {len(df)}")
                self._log(f"   • Аугментированных: {len(aug_df)}")
            
            if aug_dir:
                container.image_dir = aug_dir.parent
            
            # Добавляем метаданные в контейнер
            container.metadata['augmentation'] = {
                'original_size': len(df),
                'augmented_size': len(aug_df),
                'total_size': len(container.data),
                'augment_factor': self.augment_factor,
                'balance_classes': self.balance_classes
            }
            
            container.recommendations.append({
                "type": "success",
                "message": f"Аугментация выполнена успешно. Добавлено {len(aug_df)} изображений",
                "original_size": len(df),
                "augmented_size": len(container.data),
                "output_dir": str(aug_dir) if aug_dir else None
            })
            
        else:
            if self.verbose:
                self._log("⚠️ Аугментация не добавила новых изображений")
            # Помечаем все как не аугментированные
            df['_augmented'] = False
            container.data = df
            
            container.recommendations.append({
                "type": "warning",
                "message": "Аугментация не добавила новые изображения. Проверьте настройки.",
                "reason": "augmented_rows пустой"
            })
        
        # Сохраняем примеры аугментаций
        if self.save_examples and self._transform is not None:
            self._save_augmentation_examples(container)
        
        return container
    
    def _save_augmentation_examples(self, container: DataContainer):
        """Сохраняет примеры аугментаций для визуализации"""
        import cv2
        
        if self.examples_dir:
            examples_dir = self.examples_dir / "augmentation_examples"
            examples_dir.mkdir(parents=True, exist_ok=True)
        else:
            examples_dir = Path("augmentation_examples")
            examples_dir.mkdir(exist_ok=True)
        
        # Разделяем на оригинальные и аугментированные
        if '_augmented' in container.data.columns:
            original_df = container.data[~container.data['_augmented']]
        else:
            original_df = container.data
        
        # Берём несколько оригинальных изображений для демонстрации
        n_samples = min(self.n_examples, len(original_df))
        if n_samples == 0:
            return
        
        sample_indices = np.random.choice(len(original_df), size=n_samples, replace=False)
        
        self._examples = []
        
        for i, idx in enumerate(sample_indices):
            row = original_df.iloc[idx]
            
            if container.image_column and container.image_column in row:
                img_path = Path(row[container.image_column])
                if container.image_dir:
                    img_path = container.image_dir / img_path
                
                if img_path.exists():
                    # Загружаем оригинал
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Создаём несколько аугментаций
                        for aug_idx in range(3):
                            augmented = self._transform(image=img_rgb)["image"]
                            
                            # Сохраняем
                            filename = f"example_{i:02d}_aug{aug_idx}.jpg"
                            save_path = examples_dir / filename
                            cv2.imwrite(
                                str(save_path),
                                cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                            )
                            
                            self._examples.append({
                                "original": str(img_path),
                                "augmented": str(save_path),
                                "label": row.get(container.target_column, "") if container.target_column else "",
                                "index": i,
                                "augmentation": aug_idx
                            })
        
        # Сохраняем метаданные
        if self._examples:
            examples_df = pd.DataFrame(self._examples)
            examples_df.to_csv(examples_dir / "examples_metadata.csv", index=False)
            
            if self.verbose:
                self._log(f"📸 Сохранено {len(self._examples)} примеров аугментаций в {examples_dir}")
            
            container.recommendations.append({
                "type": "visualization",
                "message": f"Сохранено {len(self._examples)} примеров аугментаций",
                "examples_dir": str(examples_dir),
                "examples_count": len(self._examples)
            })

    def _log(self, message: str, level: str = "info"):
        """Унифицированное логирование"""
        self._logs.append(message)
        
        if not self.verbose:
            return
        
        if level == "info":
            print(f"   [ImageAugmentor] {message}")
        elif level == "warning":
            print(f"   ⚠️ [ImageAugmentor] {message}")
        elif level == "debug":
            self._logger.debug(message)
        else:
            print(f"   [{level.upper()}] [ImageAugmentor] {message}")
    
    def get_examples(self) -> list[dict]:
        """Возвращает сохранённые примеры аугментаций"""
        return self._examples
    
    def get_augmentation_stats(self) -> Dict[str, Any]:
        """Возвращает статистику аугментации"""
        return {
            'augmented_count': len(self._augmented_info),
            'logs': self._logs[-10:],  # Последние 10 логов
            'verbose': self.verbose
        }
    
    def get_augmented_info(self) -> Dict[int, Dict]:
        """Возвращает информацию об аугментированных изображениях"""
        return self._augmented_info.copy()
    
    def plot_examples(self, n: int = 4):
        """Визуализирует примеры аугментаций"""
        if not self._examples:
            print("Примеры не сохранены. Установите save_examples=True")
            return
        
        import matplotlib.pyplot as plt
        import cv2
        
        # Группируем по оригинальным изображениям
        examples_by_original = {}
        for ex in self._examples:
            key = ex['original']
            if key not in examples_by_original:
                examples_by_original[key] = []
            examples_by_original[key].append(ex)
        
        n = min(n, len(examples_by_original))
        orig_keys = list(examples_by_original.keys())[:n]
        
        fig, axes = plt.subplots(n, 4, figsize=(15, 3 * n))
        
        if n == 1:
            axes = axes.reshape(1, -1)
        
        for i, orig_key in enumerate(orig_keys):
            orig_examples = examples_by_original[orig_key]
            orig_example = orig_examples[0]
            
            # Оригинал
            orig_img = cv2.imread(orig_example["original"])
            if orig_img is not None:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                
                axes[i, 0].imshow(orig_img)
                axes[i, 0].set_title(f"Оригинал\n{orig_example['label']}")
                axes[i, 0].axis('off')
                
                # 3 аугментации
                for j in range(3):
                    if j < len(orig_examples):
                        aug_example = orig_examples[j]
                        aug_img = cv2.imread(aug_example["augmented"])
                        if aug_img is not None:
                            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                            
                            axes[i, j + 1].imshow(aug_img)
                            axes[i, j + 1].set_title(f"Аугментация {j + 1}")
                            axes[i, j + 1].axis('off')
        
        plt.suptitle("Примеры аугментаций изображений", fontsize=16)
        plt.tight_layout()
        plt.savefig("augmentation_examples_plot.jpg", dpi=150, bbox_inches='tight')
        plt.show()
    
    def _augment_class(
        self,
        df: pd.DataFrame,
        image_col: str,
        image_dir: Path | None,
        output_dir: Path | None,
        n_samples: int,
        counter: int,
        original_df: Optional[pd.DataFrame] = None  # ← ДОБАВИЛ ОПЦИОНАЛЬНЫЙ ПАРАМЕТР
    ) -> tuple[list[dict], int]:
        """Аугментация для одного класса с отслеживанием"""
        import cv2
        
        augmented = []
        
        if len(df) == 0:
            return augmented, counter
        
        indices = np.random.choice(len(df), size=n_samples, replace=True)
        
        for i, idx in enumerate(indices):
            row = df.iloc[idx].to_dict()
            original_idx = df.index[idx]
            
            # Загружаем изображение
            img_path = Path(row[image_col])
            if image_dir:
                img_path = image_dir / img_path
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Конвертируем BGR -> RGB для albumentations
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Применяем аугментации
            augmented_img = self._transform(image=img)["image"]
            
            # Конвертируем обратно RGB -> BGR
            augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)
            
            # Создаём новую строку
            new_row = row.copy()
            
            # ОБЯЗАТЕЛЬНО добавляем source_idx для отслеживания оригинала
            new_row["_source_idx"] = original_idx
            
            # Сохраняем информацию об аугментации
            aug_info = {
                'original_index': int(original_idx),
                'original_path': str(img_path),
                'transformations': 'random_augmentation',
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Сохраняем
            if output_dir:
                aug_filename = f"aug_{counter:06d}.jpg"
                aug_path = output_dir / aug_filename
                cv2.imwrite(str(aug_path), augmented_img)
                
                relative_path = f"augmented/{aug_filename}"
                new_row[image_col] = relative_path
            
                new_row["_augmented"] = True
                augmented.append(new_row)
                
                # Сохраняем путь к аугментированному файлу
                aug_info['augmented_path'] = str(aug_path)
                
                # Запоминаем информацию об аугментации
                if original_df is not None:
                    # Вычисляем индекс в будущем объединённом DataFrame
                    total_original_len = len(original_df)
                    current_aug_count = len(augmented)
                    # Будем вычислять окончательный индекс после добавления
                    self._augmented_info[len(original_df) + len(augmented) - 1] = aug_info
                
                counter += 1
            else:
                # Без сохранения — храним в памяти (для небольших датасетов)
                new_row["_augmented_image"] = augmented_img
                new_row["_augmented"] = True
                augmented.append(new_row)
                
                # Запоминаем информацию об аугментации
                if original_df is not None:
                    self._augmented_info[len(original_df) + len(augmented) - 1] = aug_info
                
                counter += 1
        
        return augmented, counter
    
    def get_albumentations_pipeline(self) -> Any:
        """Получить Albumentations pipeline для инференса"""
        return self._transform
