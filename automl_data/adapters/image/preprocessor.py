# automl_data/adapters/image/preprocessor.py
"""
Препроцессор изображений.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from ..base import BaseAdapter
from ...core.container import DataContainer, ProcessingStage
from ...core.config import ImageConfig
from ...utils.decorators import safe_transform
from ...utils.dependencies import require_package


class ImagePreprocessor(BaseAdapter):
    """
    Препроцессинг изображений.
    
    Включает:
    - Ресайз до целевого размера
    - Нормализация (ImageNet статистики)
    - Удаление битых файлов
    
    Example:
        >>> preprocessor = ImagePreprocessor(target_size=(224, 224))
        >>> result = preprocessor.fit_transform(container)
    """
    
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    def __init__(
        self,
        config: ImageConfig | None = None,
        target_size: tuple[int, int] = (224, 224),
        normalize: bool = True,
        keep_aspect_ratio: bool = True,
        output_dir: Path | None = None,
        **kwargs
    ):
        super().__init__(name="ImagePreprocessor", **kwargs)
        
        if config:
            self.target_size = config.target_size
            self.normalize = config.normalize
            self.mean = config.mean
            self.std = config.std
            self.keep_aspect_ratio = config.keep_aspect_ratio
        else:
            self.target_size = target_size
            self.normalize = normalize
            self.mean = self.IMAGENET_MEAN
            self.std = self.IMAGENET_STD
            self.keep_aspect_ratio = keep_aspect_ratio
        
        self.output_dir = Path(output_dir) if output_dir else None
        self._valid_indices: list[int] = []
    
    def _fit_impl(self, container: DataContainer) -> None:
        """Проверяем какие изображения валидны"""
        require_package("cv2", "opencv-python")
        import cv2
        
        paths = container.image_paths or []
        self._valid_indices = []
        invalid_count = 0
        
        for idx, path in enumerate(paths):
            try:
                img = cv2.imread(str(path))
                if img is not None:
                    self._valid_indices.append(idx)
                else:
                    invalid_count += 1
            except (IOError, OSError) as e:
                self._logger.debug(f"Cannot read {path}: {e}")
                invalid_count += 1
        
        self._fit_info.update({
            "valid_images": len(self._valid_indices),
            "invalid_images": invalid_count,
            "target_size": self.target_size
        })
        
        if invalid_count > 0:
            self._logger.warning(f"Found {invalid_count} invalid images")
    
    @safe_transform(preserve_target=True, sync_state=True, reset_index=True)
    def _transform_impl(self, container: DataContainer) -> DataContainer:
        if not container.image_column:
            return container
        
        require_package("cv2", "opencv-python")
        import cv2
        
        df = container.data.copy()
        
        # Фильтруем невалидные
        if self._valid_indices:
            df = df.iloc[self._valid_indices]
        
        # Сохраняем обработанные изображения
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            new_paths = self._process_and_save(df, container, cv2)
            df[container.image_column] = new_paths[:len(df)]
            container.image_dir = None
        
        container.data = df
        container.stage = ProcessingStage.CLEANED
        
        container.recommendations.append({
            "type": "image_preprocessing",
            "valid_images": len(self._valid_indices),
            "target_size": self.target_size
        })
        
        return container
    
    def _process_and_save(self, df, container, cv2) -> list[str]:
        """Обработка и сохранение изображений"""
        new_paths = []
        
        for idx, row in df.iterrows():
            old_path = Path(row[container.image_column])
            if container.image_dir:
                old_path = container.image_dir / old_path
            
            img = cv2.imread(str(old_path))
            if img is None:
                continue
            
            img = self._resize(img, cv2)
            
            new_path = self.output_dir / f"{idx:06d}.jpg"
            cv2.imwrite(str(new_path), img)
            new_paths.append(str(new_path))
        
        return new_paths
    
    def _resize(self, img: np.ndarray, cv2) -> np.ndarray:
        """Ресайз с сохранением пропорций"""
        h, w = img.shape[:2]
        target_w, target_h = self.target_size
        
        if self.keep_aspect_ratio:
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            img = cv2.resize(img, (new_w, new_h))
            
            # Padding
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            
            img = cv2.copyMakeBorder(
                img,
                pad_h // 2, pad_h - pad_h // 2,
                pad_w // 2, pad_w - pad_w // 2,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )
        else:
            img = cv2.resize(img, self.target_size)
        
        return img
    
    def get_transform_pipeline(self) -> Any:
        """Получить torchvision transforms для инференса"""
        require_package("torchvision", "torchvision")
        from torchvision import transforms
        
        transform_list = [
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        return transforms.Compose(transform_list)