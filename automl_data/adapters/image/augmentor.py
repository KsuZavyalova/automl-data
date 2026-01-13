# automl_data/adapters/image/augmentor.py
"""
Аугментация изображений с Albumentations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from ..base import BaseAdapter
from ...core.container import DataContainer, ProcessingStage
from ...core.config import ImageConfig
from ...utils.decorators import safe_transform
from ...utils.dependencies import require_package


class ImageAugmentor(BaseAdapter):
    """
    Аугментация изображений через Albumentations.
    
    Поддерживает:
    - Геометрические трансформации (flip, rotate, zoom)
    - Цветовые аугментации (brightness, contrast)
    - RandAugment-style трансформации
    - Балансировку классов через аугментацию
    
    Example:
        >>> augmentor = ImageAugmentor(augment_factor=3.0, balance_classes=True)
        >>> result = augmentor.fit_transform(container)
    """
    
    def __init__(
        self,
        config: ImageConfig | None = None,
        augment_factor: float = 3.0,
        balance_classes: bool = True,
        
        # Геометрические
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        rotation_range: int = 15,
        zoom_range: tuple[float, float] = (0.9, 1.1),
        shift_range: float = 0.1,
        
        # Цветовые
        brightness_range: tuple[float, float] = (0.8, 1.2),
        contrast_range: tuple[float, float] = (0.8, 1.2),
        
        # Продвинутые
        use_randaugment: bool = True,
        use_cutout: bool = True,
        add_noise: bool = True,
        add_blur: bool = True,
        
        output_dir: Path | None = None,
        random_state: int = 42,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__(name="ImageAugmentor", **kwargs)
        
        self.verbose = verbose
        
        if config:
            self._init_from_config(config)
        else:
            self.augment_factor = augment_factor
            self.balance_classes = balance_classes
            self.horizontal_flip = horizontal_flip
            self.vertical_flip = vertical_flip
            self.rotation_range = rotation_range
            self.zoom_range = zoom_range
            self.shift_range = shift_range
            self.brightness_range = brightness_range
            self.contrast_range = contrast_range
            self.use_randaugment = use_randaugment
        
        self.use_cutout = use_cutout
        self.add_noise = add_noise
        self.add_blur = add_blur
        self.output_dir = Path(output_dir) if output_dir else None
        self.random_state = random_state
        
        self._transform = None
        self._class_counts: dict = {}
        self._target_count: int = 0
        
        np.random.seed(self.random_state)
    
    def _init_from_config(self, config: ImageConfig):
        """Инициализация из конфига"""
        self.augment_factor = config.augment_factor
        self.balance_classes = config.balance_classes
        self.horizontal_flip = config.horizontal_flip
        self.vertical_flip = getattr(config, 'vertical_flip', False)
        self.rotation_range = config.rotation_range
        self.zoom_range = config.zoom_range
        self.shift_range = getattr(config, 'shift_range', 0.1)
        self.brightness_range = config.brightness_range
        self.contrast_range = config.contrast_range
        self.use_randaugment = config.use_randaugment
    
    def _fit_impl(self, container: DataContainer) -> None:
        require_package("albumentations", "albumentations")
        import albumentations as A
        
        np.random.seed(self.random_state)
        
        self._transform = self._build_transform_pipeline(A)
        
        if self.balance_classes and container.target_column:
            self._class_counts = container.data[container.target_column].value_counts().to_dict()
            self._target_count = max(self._class_counts.values())
            
            if self.verbose:
                self._logger.info(f"Class distribution: {self._class_counts}")
        
        self._fit_info.update({
            "augment_factor": self.augment_factor,
            "balance_classes": self.balance_classes,
            "class_counts": self._class_counts
        })
    
    def _build_transform_pipeline(self, A):
        """Создание Albumentations pipeline"""
        transforms = []
        
        # Геометрические
        if self.horizontal_flip:
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if self.vertical_flip:
            transforms.append(A.VerticalFlip(p=0.5))
        
        if self.rotation_range > 0:
            transforms.append(A.Rotate(limit=self.rotation_range, p=0.5, border_mode=0))
        
        if self.shift_range > 0 or self.zoom_range != (1.0, 1.0):
            transforms.append(A.ShiftScaleRotate(
                shift_limit=self.shift_range,
                scale_limit=(self.zoom_range[0] - 1, self.zoom_range[1] - 1),
                rotate_limit=0,
                p=0.5,
                border_mode=0
            ))
        
        # Цветовые
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=(self.brightness_range[0] - 1, self.brightness_range[1] - 1),
            contrast_limit=(self.contrast_range[0] - 1, self.contrast_range[1] - 1),
            p=0.5
        ))
        
        if self.use_randaugment:
            transforms.append(A.OneOf([
                A.Equalize(p=1.0),
                A.Posterize(p=1.0),
                A.Sharpen(p=1.0),
            ], p=0.3))
        
        # Шум и блюр
        if self.add_noise:
            transforms.append(A.GaussNoise(var_limit=(10, 50), p=0.2))
        
        if self.add_blur:
            transforms.append(A.GaussianBlur(blur_limit=3, p=0.2))
        
        if self.use_cutout:
            transforms.append(A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32,
                min_holes=1, min_height=8, min_width=8,
                fill_value=0, p=0.3
            ))
        
        return A.Compose(transforms)
    
    @safe_transform(preserve_target=False, sync_state=True, reset_index=True)
    def _transform_impl(self, container: DataContainer) -> DataContainer:
        if not container.image_column or self._transform is None:
            return container
        
        require_package("cv2", "opencv-python")
        import cv2
        
        df = container.data.copy()
        image_col = container.image_column
        target_col = container.target_column
        
        # Создаём директорию
        aug_dir = None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            aug_dir = self.output_dir / "augmented"
            aug_dir.mkdir(exist_ok=True)
        
        # Генерируем аугментации
        augmented_rows = self._generate_augmentations(
            df, container, cv2, aug_dir
        )
        
        # Объединяем
        if augmented_rows:
            df['_augmented'] = False
            aug_df = pd.DataFrame(augmented_rows)
            aug_df['_augmented'] = True
            
            container.data = pd.concat([df, aug_df], ignore_index=True)
            container.stage = ProcessingStage.AUGMENTED
            
            if aug_dir:
                container.image_dir = aug_dir.parent
            
            container.recommendations.append({
                "type": "image_augmentation",
                "original_size": len(df),
                "augmented_size": len(container.data),
                "new_images": len(aug_df)
            })
            
            if self.verbose:
                self._logger.info(
                    f"Augmented: {len(df)} -> {len(container.data)} images"
                )
        else:
            df['_augmented'] = False
            container.data = df
        
        return container
    
    def _generate_augmentations(
        self,
        df: pd.DataFrame,
        container: DataContainer,
        cv2,
        aug_dir: Path | None
    ) -> list[dict]:
        """Генерация аугментированных изображений"""
        augmented = []
        counter = 0
        
        image_col = container.image_column
        target_col = container.target_column
        image_dir = container.image_dir
        
        if self.balance_classes and target_col and self._class_counts:
            for label, count in self._class_counts.items():
                class_df = df[df[target_col] == label]
                n_to_generate = self._target_count - count
                
                if n_to_generate > 0:
                    new_rows, counter = self._augment_samples(
                        class_df, image_col, image_dir, aug_dir,
                        n_to_generate, counter, cv2
                    )
                    augmented.extend(new_rows)
        else:
            n_to_generate = max(0, int(len(df) * (self.augment_factor - 1)))
            if n_to_generate > 0:
                new_rows, counter = self._augment_samples(
                    df, image_col, image_dir, aug_dir,
                    n_to_generate, counter, cv2
                )
                augmented.extend(new_rows)
        
        return augmented
    
    def _augment_samples(
        self,
        df: pd.DataFrame,
        image_col: str,
        image_dir: Path | None,
        output_dir: Path | None,
        n_samples: int,
        counter: int,
        cv2
    ) -> tuple[list[dict], int]:
        """Аугментация сэмплов"""
        augmented = []
        
        if len(df) == 0:
            return augmented, counter
        
        indices = np.random.choice(len(df), size=n_samples, replace=True)
        
        for idx in indices:
            row = df.iloc[idx].to_dict()
            
            img_path = Path(row[image_col])
            if image_dir:
                img_path = image_dir / img_path
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented_img = self._transform(image=img_rgb)["image"]
            augmented_bgr = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)
            
            new_row = row.copy()
            
            if output_dir:
                aug_filename = f"aug_{counter:06d}.jpg"
                aug_path = output_dir / aug_filename
                cv2.imwrite(str(aug_path), augmented_bgr)
                new_row[image_col] = f"augmented/{aug_filename}"
            
            augmented.append(new_row)
            counter += 1
        
        return augmented, counter
    
    def get_albumentations_pipeline(self) -> Any:
        """Получить Albumentations pipeline"""
        return self._transform