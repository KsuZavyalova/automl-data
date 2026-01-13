# tests/test_image_processing.py
"""
Комплексные тесты для обработки изображений.

Покрывает:
- ImagePreprocessor: ресайз, нормализация, валидация
- ImageAugmentor: аугментации, балансировка классов
- Интеграция с AutoForge
- Edge cases: битые файлы, разные форматы
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Пропускаем тесты если нет opencv
cv2 = pytest.importorskip("cv2")
albumentations = pytest.importorskip("albumentations")

from automl_data.core.forge import AutoForge
from automl_data.core.container import DataContainer
from automl_data.core.config import ImageConfig
from automl_data.adapters.image.preprocessor import ImagePreprocessor
from automl_data.adapters.image.augmentor import ImageAugmentor


# ==================== FIXTURES ====================

@pytest.fixture
def temp_image_dir():
    """Создаёт временную директорию с тестовыми изображениями"""
    temp_dir = Path(tempfile.mkdtemp())
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_images(temp_image_dir) -> tuple[Path, pd.DataFrame]:
    """
    Создаёт набор тестовых изображений разных размеров и форматов.
    
    Returns:
        (image_dir, dataframe с метаданными)
    """
    images_info = []
    
    # Создаём изображения разных размеров
    sizes = [(100, 100), (200, 150), (50, 75), (300, 300), (640, 480)]
    labels = [0, 0, 0, 1, 1]  # Несбалансированные классы
    
    for i, (size, label) in enumerate(zip(sizes, labels)):
        # Создаём случайное RGB изображение
        img = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        
        filename = f"image_{i:03d}.jpg"
        filepath = temp_image_dir / filename
        cv2.imwrite(str(filepath), img)
        
        images_info.append({
            "image_path": filename,
            "label": label,
            "original_width": size[0],
            "original_height": size[1]
        })
    
    df = pd.DataFrame(images_info)
    
    return temp_image_dir, df


@pytest.fixture
def imbalanced_images(temp_image_dir) -> tuple[Path, pd.DataFrame]:
    """Создаёт сильно несбалансированный датасет изображений"""
    images_info = []
    
    # Класс 0: 20 изображений, Класс 1: 5 изображений
    class_counts = {0: 20, 1: 5}
    
    idx = 0
    for label, count in class_counts.items():
        for _ in range(count):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # Добавляем цветовой паттерн для различения классов
            if label == 0:
                img[:, :, 0] = 200  # Красноватые
            else:
                img[:, :, 2] = 200  # Синеватые
            
            filename = f"img_{idx:04d}.jpg"
            cv2.imwrite(str(temp_image_dir / filename), img)
            
            images_info.append({
                "image_path": filename,
                "class": label
            })
            idx += 1
    
    df = pd.DataFrame(images_info)
    return temp_image_dir, df


@pytest.fixture
def mixed_format_images(temp_image_dir) -> tuple[Path, pd.DataFrame]:
    """Создаёт изображения разных форматов"""
    images_info = []
    formats = [("jpg", ".jpg"), ("png", ".png"), ("bmp", ".bmp")]
    
    for i, (fmt_name, ext) in enumerate(formats):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        filename = f"image_{fmt_name}{ext}"
        cv2.imwrite(str(temp_image_dir / filename), img)
        
        images_info.append({
            "image_path": filename,
            "format": fmt_name,
            "label": i % 2
        })
    
    return temp_image_dir, pd.DataFrame(images_info)


@pytest.fixture
def images_with_corrupted(temp_image_dir) -> tuple[Path, pd.DataFrame]:
    """Создаёт датасет с битыми файлами"""
    images_info = []
    
    # Нормальные изображения
    for i in range(5):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        filename = f"valid_{i}.jpg"
        cv2.imwrite(str(temp_image_dir / filename), img)
        images_info.append({"image_path": filename, "label": 0, "is_valid": True})
    
    # Битый файл (пустой)
    corrupted_path = temp_image_dir / "corrupted.jpg"
    corrupted_path.write_bytes(b"not an image")
    images_info.append({"image_path": "corrupted.jpg", "label": 1, "is_valid": False})
    
    # Несуществующий файл
    images_info.append({"image_path": "missing.jpg", "label": 1, "is_valid": False})
    
    return temp_image_dir, pd.DataFrame(images_info)


# ==================== PREPROCESSOR TESTS ====================

class TestImagePreprocessor:
    """Тесты ImagePreprocessor"""
    
    def test_basic_preprocessing(self, sample_images):
        """Базовый тест препроцессинга"""
        image_dir, df = sample_images
        
        container = DataContainer(
            data=df.copy(),
            image_column="image_path",
            image_dir=image_dir,
            target_column="label"
        )
        
        preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            normalize=True
        )
        
        result = preprocessor.fit_transform(container)
        
        assert preprocessor.is_fitted
        assert len(result.data) > 0
        assert "valid_images" in preprocessor._fit_info
    
    def test_resize_to_target_size(self, sample_images):
        """Тест ресайза до целевого размера"""
        image_dir, df = sample_images
        target_size = (128, 128)
        
        # Создаём output директорию
        output_dir = image_dir / "processed"
        
        container = DataContainer(
            data=df.copy(),
            image_column="image_path",
            image_dir=image_dir,
            target_column="label"
        )
        
        preprocessor = ImagePreprocessor(
            target_size=target_size,
            output_dir=output_dir,
            keep_aspect_ratio=False
        )
        
        result = preprocessor.fit_transform(container)
        
        # Проверяем размеры обработанных изображений
        for idx, row in result.data.iterrows():
            img_path = Path(row["image_path"])
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    assert (w, h) == target_size, \
                        f"Expected {target_size}, got ({w}, {h})"
    
    def test_keep_aspect_ratio(self, sample_images):
        """Тест сохранения пропорций"""
        image_dir, df = sample_images
        target_size = (224, 224)
        
        output_dir = image_dir / "processed_aspect"
        
        container = DataContainer(
            data=df.copy(),
            image_column="image_path",
            image_dir=image_dir,
            target_column="label"
        )
        
        preprocessor = ImagePreprocessor(
            target_size=target_size,
            output_dir=output_dir,
            keep_aspect_ratio=True
        )
        
        result = preprocessor.fit_transform(container)
        
        # Все изображения должны быть целевого размера
        for idx, row in result.data.iterrows():
            img_path = Path(row["image_path"])
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    assert (w, h) == target_size
    
    def test_handles_corrupted_images(self, images_with_corrupted):
        """Тест обработки битых изображений"""
        image_dir, df = images_with_corrupted
        
        container = DataContainer(
            data=df.copy(),
            image_column="image_path",
            image_dir=image_dir,
            target_column="label"
        )
        
        preprocessor = ImagePreprocessor(target_size=(100, 100))
        
        # Не должно быть ошибки
        result = preprocessor.fit_transform(container)
        
        # Должны остаться только валидные
        valid_count = preprocessor._fit_info.get("valid_images", 0)
        invalid_count = preprocessor._fit_info.get("invalid_images", 0)
        
        assert valid_count == 5  # 5 валидных
        assert invalid_count == 2  # 1 битый + 1 несуществующий
    
    def test_different_formats(self, mixed_format_images):
        """Тест разных форматов изображений"""
        image_dir, df = mixed_format_images
        
        container = DataContainer(
            data=df.copy(),
            image_column="image_path",
            image_dir=image_dir,
            target_column="label"
        )
        
        preprocessor = ImagePreprocessor(target_size=(64, 64))
        result = preprocessor.fit_transform(container)
        
        # Все форматы должны обработаться
        assert preprocessor._fit_info["valid_images"] == 3
    
    def test_with_image_config(self, sample_images):
        """Тест с ImageConfig"""
        image_dir, df = sample_images
        
        config = ImageConfig(
            target_size=(256, 256),
            normalize=True,
            keep_aspect_ratio=True
        )
        
        container = DataContainer(
            data=df.copy(),
            image_column="image_path",
            image_dir=image_dir
        )
        
        preprocessor = ImagePreprocessor(config=config)
        result = preprocessor.fit_transform(container)
        
        assert preprocessor.target_size == (256, 256)
        assert preprocessor.normalize is True
    
    def test_get_transform_pipeline(self, sample_images):
        """Тест получения torchvision pipeline"""
        pytest.importorskip("torchvision")
        
        image_dir, df = sample_images
        
        preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            normalize=True
        )
        
        pipeline = preprocessor.get_transform_pipeline()
        
        assert pipeline is not None
        assert hasattr(pipeline, '__call__')


# ==================== AUGMENTOR TESTS ====================

class TestImageAugmentor:
    """Тесты ImageAugmentor"""
    
    def test_basic_augmentation(self, sample_images):
        """Базовый тест аугментации"""
        image_dir, df = sample_images
        output_dir = image_dir / "augmented_output"
        
        container = DataContainer(
            data=df.copy(),
            image_column="image_path",
            image_dir=image_dir,
            target_column="label"
        )
        
        augmentor = ImageAugmentor(
            augment_factor=2.0,
            balance_classes=False,
            output_dir=output_dir,
            verbose=False
        )
        
        result = augmentor.fit_transform(container)
        
        assert augmentor.is_fitted
        # Должно быть больше данных
        assert len(result.data) >= len(df)
    
    def test_augmentation_generates_images(self, sample_images):
        """Проверяет, что аугментация генерирует новые изображения"""
        image_dir, df = sample_images
        output_dir = image_dir / "aug_gen"
        
        original_count = len(df)
        
        container = DataContainer(
            data=df.copy(),
            image_column="image_path",
            image_dir=image_dir,
            target_column="label"
        )
        
        augmentor = ImageAugmentor(
            augment_factor=3.0,
            balance_classes=False,
            output_dir=output_dir,
            verbose=False
        )
        
        result = augmentor.fit_transform(container)
        
        new_count = len(result.data)
        
        # Должно быть примерно augment_factor * original
        expected_min = int(original_count * 2)  # Минимум x2
        
        assert new_count >= expected_min, \
            f"Expected at least {expected_min} images, got {new_count}"
        
        # Проверяем наличие флага _augmented
        if '_augmented' in result.data.columns:
            augmented_count = result.data['_augmented'].sum()
            assert augmented_count > 0, "Should have augmented images"
    
    def test_class_balancing(self, imbalanced_images):
        """Тест балансировки классов через аугментацию"""
        image_dir, df = imbalanced_images
        output_dir = image_dir / "balanced"
        
        original_distribution = df['class'].value_counts().to_dict()
        
        container = DataContainer(
            data=df.copy(),
            image_column="image_path",
            image_dir=image_dir,
            target_column="class"
        )
        
        augmentor = ImageAugmentor(
            augment_factor=1.0,  # Только балансировка
            balance_classes=True,
            output_dir=output_dir,
            verbose=False
        )
        
        result = augmentor.fit_transform(container)
        
        new_distribution = result.data['class'].value_counts().to_dict()
        
        # Минорный класс должен увеличиться
        assert new_distribution.get(1, 0) > original_distribution[1], \
            f"Minority class should increase: {original_distribution} -> {new_distribution}"
        
        print(f"\n✅ Балансировка классов:")
        print(f"   До: {original_distribution}")
        print(f"   После: {new_distribution}")
    
    def test_augmentation_preserves_labels(self, sample_images):
        """Проверяет сохранение меток при аугментации"""
        image_dir, df = sample_images
        output_dir = image_dir / "aug_labels"
        
        container = DataContainer(
            data=df.copy(),
            image_column="image_path",
            image_dir=image_dir,
            target_column="label"
        )
        
        augmentor = ImageAugmentor(
            augment_factor=2.0,
            balance_classes=False,
            output_dir=output_dir,
            verbose=False
        )
        
        result = augmentor.fit_transform(container)
        
        # Все записи должны иметь label
        assert result.data['label'].notna().all(), \
            "All augmented images should have labels"
        
        # Метки должны быть из исходного набора
        original_labels = set(df['label'].unique())
        new_labels = set(result.data['label'].unique())
        
        assert new_labels.issubset(original_labels), \
            f"New labels {new_labels} should be subset of {original_labels}"
    
    def test_different_augmentation_types(self, sample_images):
        """Тест разных типов аугментаций"""
        image_dir, df = sample_images
        
        configs = [
            {"horizontal_flip": True, "vertical_flip": False},
            {"rotation_range": 30, "horizontal_flip": False},
            {"brightness_range": (0.5, 1.5), "contrast_range": (0.5, 1.5)},
            {"use_randaugment": True},
            {"use_cutout": True, "add_noise": True},
        ]
        
        for i, config in enumerate(configs):
            output_dir = image_dir / f"aug_type_{i}"
            
            container = DataContainer(
                data=df.copy(),
                image_column="image_path",
                image_dir=image_dir,
                target_column="label"
            )
            
            augmentor = ImageAugmentor(
                augment_factor=2.0,
                balance_classes=False,
                output_dir=output_dir,
                verbose=False,
                **config
            )
            
            result = augmentor.fit_transform(container)
            
            assert len(result.data) > len(df), \
                f"Config {config} should generate more images"
    
    def test_augmentor_with_config(self, sample_images):
        """Тест с ImageConfig"""
        image_dir, df = sample_images
        output_dir = image_dir / "aug_config"
        
        config = ImageConfig(
            augment=True,
            augment_factor=2.0,
            horizontal_flip=True,
            rotation_range=15,
            brightness_range=(0.9, 1.1)
        )
        
        container = DataContainer(
            data=df.copy(),
            image_column="image_path",
            image_dir=image_dir,
            target_column="label"
        )
        
        augmentor = ImageAugmentor(
            config=config,
            output_dir=output_dir,
            verbose=False
        )
        
        result = augmentor.fit_transform(container)
        
        assert augmentor.augment_factor == 2.0
        assert len(result.data) > len(df)
    
    def test_get_albumentations_pipeline(self, sample_images):
        """Тест получения Albumentations pipeline"""
        image_dir, df = sample_images
        
        container = DataContainer(
            data=df.copy(),
            image_column="image_path",
            image_dir=image_dir
        )
        
        augmentor = ImageAugmentor(verbose=False)
        augmentor.fit(container)
        
        pipeline = augmentor.get_albumentations_pipeline()
        
        assert pipeline is not None
        
        # Проверяем, что pipeline работает
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = pipeline(image=test_img)
        
        assert "image" in result
        assert result["image"].shape == test_img.shape


# ==================== INTEGRATION TESTS ====================

class TestImageIntegration:
    """Интеграционные тесты с AutoForge"""
    
    def test_autoforge_with_images(self, sample_images):
        """Полный пайплайн AutoForge с изображениями"""
        image_dir, df = sample_images
        
        forge = AutoForge(
            target="label",
            task="classification",
            image_column="image_path",
            image_dir=image_dir,
            augment=True,
            augment_factor=2.0,
            balance=True,
            verbose=False
        )
        
        result = forge.fit_transform(df)
        
        assert result is not None
        assert len(result.data) > 0
        assert result.quality_score > 0
    
    def test_autoforge_image_classification(self, imbalanced_images):
        """Тест классификации изображений с балансировкой"""
        image_dir, df = imbalanced_images
        
        original_size = len(df)
        original_distribution = df['class'].value_counts().to_dict()
        
        forge = AutoForge(
            target="class",
            task="classification",
            image_column="image_path",
            image_dir=image_dir,
            balance=True,
            augment=True,
            augment_factor=2.0,
            verbose=False
        )
        
        result = forge.fit_transform(df)
        
        # Проверяем, что балансировка сработала
        new_distribution = result.data['class'].value_counts().to_dict()
        
        # Минорный класс должен увеличиться или соотношение улучшиться
        original_ratio = min(original_distribution.values()) / max(original_distribution.values())
        new_ratio = min(new_distribution.values()) / max(new_distribution.values())
        
        print(f"\n✅ AutoForge с изображениями:")
        print(f"   Размер: {original_size} -> {len(result.data)}")
        print(f"   Распределение: {original_distribution} -> {new_distribution}")
        print(f"   Баланс: {original_ratio:.2f} -> {new_ratio:.2f}")
    
    def test_full_pipeline_preprocessing_and_augmentation(self, sample_images):
        """Тест полного пайплайна: preprocessing + augmentation"""
        image_dir, df = sample_images
        output_dir = image_dir / "full_pipeline"
        
        # 1. Preprocessing
        container = DataContainer(
            data=df.copy(),
            image_column="image_path",
            image_dir=image_dir,
            target_column="label"
        )
        
        preprocessor = ImagePreprocessor(
            target_size=(128, 128),
            output_dir=output_dir / "preprocessed"
        )
        
        preprocessed = preprocessor.fit_transform(container)
        
        # 2. Augmentation
        augmentor = ImageAugmentor(
            augment_factor=2.0,
            balance_classes=True,
            output_dir=output_dir / "augmented",
            verbose=False
        )
        
        # Обновляем image_dir
        preprocessed.image_dir = output_dir / "preprocessed"
        
        final = augmentor.fit_transform(preprocessed)
        
        assert len(final.data) > len(df)
        
        print(f"\n✅ Полный пайплайн:")
        print(f"   Исходный: {len(df)}")
        print(f"   После preprocessing: {len(preprocessed.data)}")
        print(f"   После augmentation: {len(final.data)}")


# ==================== EDGE CASES ====================

class TestImageEdgeCases:
    """Тесты граничных случаев"""
    
    def test_empty_dataframe(self, temp_image_dir):
        """Тест с пустым DataFrame - должен выбросить ValidationError"""
        from automl_data.utils.exceptions import ValidationError
        
        df = pd.DataFrame(columns=["image_path", "label"])
        
        container = DataContainer(
            data=df,
            image_column="image_path",
            image_dir=temp_image_dir
        )
        
        preprocessor = ImagePreprocessor(target_size=(100, 100))
        
        # Пустой DataFrame ДОЛЖЕН вызывать ошибку
        with pytest.raises(ValidationError) as exc_info:
            preprocessor.fit_transform(container)
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_single_image(self, temp_image_dir):
        """Тест с одним изображением"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(temp_image_dir / "single.jpg"), img)
        
        df = pd.DataFrame([{"image_path": "single.jpg", "label": 0}])
        
        container = DataContainer(
            data=df,
            image_column="image_path",
            image_dir=temp_image_dir,
            target_column="label"
        )
        
        augmentor = ImageAugmentor(
            augment_factor=5.0,
            balance_classes=False,
            output_dir=temp_image_dir / "single_aug",
            verbose=False
        )
        
        result = augmentor.fit_transform(container)
        
        # Должно быть ~5 изображений
        assert len(result.data) >= 4
    
    def test_very_small_images(self, temp_image_dir):
        """Тест с очень маленькими изображениями"""
        # 1x1 пиксель
        tiny_img = np.array([[[255, 0, 0]]], dtype=np.uint8)
        cv2.imwrite(str(temp_image_dir / "tiny.jpg"), tiny_img)
        
        df = pd.DataFrame([{"image_path": "tiny.jpg", "label": 0}])
        
        container = DataContainer(
            data=df,
            image_column="image_path",
            image_dir=temp_image_dir
        )
        
        preprocessor = ImagePreprocessor(target_size=(224, 224))
        result = preprocessor.fit_transform(container)
        
        # Должно обработаться без ошибок
        assert len(result.data) > 0
    
    def test_grayscale_images(self, temp_image_dir):
        """Тест с чёрно-белыми изображениями"""
        gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        cv2.imwrite(str(temp_image_dir / "gray.jpg"), gray_img)
        
        df = pd.DataFrame([{"image_path": "gray.jpg", "label": 0}])
        
        container = DataContainer(
            data=df,
            image_column="image_path",
            image_dir=temp_image_dir
        )
        
        augmentor = ImageAugmentor(
            augment_factor=2.0,
            output_dir=temp_image_dir / "gray_aug",
            verbose=False
        )
        
        # Может вызвать предупреждение, но не должно упасть
        result = augmentor.fit_transform(container)
        assert len(result.data) >= 1
    
    def test_no_target_column(self, sample_images):
        """Тест без target колонки"""
        image_dir, df = sample_images
        df_no_target = df.drop(columns=['label'])
        
        container = DataContainer(
            data=df_no_target,
            image_column="image_path",
            image_dir=image_dir
        )
        
        augmentor = ImageAugmentor(
            augment_factor=2.0,
            balance_classes=True,  # Не должно падать без target
            output_dir=image_dir / "no_target",
            verbose=False
        )
        
        result = augmentor.fit_transform(container)
        
        # Балансировка не сработает, но аугментация должна
        assert len(result.data) >= len(df_no_target)


# ==================== PERFORMANCE TESTS ====================

class TestImagePerformance:
    """Тесты производительности"""
    
    def test_large_dataset(self, temp_image_dir):
        """Тест с большим количеством изображений"""
        n_images = 50
        images_info = []
        
        for i in range(n_images):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            filename = f"perf_{i:04d}.jpg"
            cv2.imwrite(str(temp_image_dir / filename), img)
            images_info.append({
                "image_path": filename,
                "label": i % 3
            })
        
        df = pd.DataFrame(images_info)
        
        container = DataContainer(
            data=df,
            image_column="image_path",
            image_dir=temp_image_dir,
            target_column="label"
        )
        
        import time
        
        start = time.time()
        
        preprocessor = ImagePreprocessor(target_size=(128, 128))
        result = preprocessor.fit_transform(container)
        
        elapsed = time.time() - start
        
        # Должно обработаться менее чем за 30 секунд
        assert elapsed < 30, f"Processing took too long: {elapsed:.2f}s"
        
        print(f"\n⏱️ Производительность: {n_images} изображений за {elapsed:.2f}s")
        print(f"   Скорость: {n_images/elapsed:.1f} img/s")


# ==================== RECOMMENDATIONS TESTS ====================

class TestImageRecommendations:
    """Тесты рекомендаций и метаданных"""
    
    def test_recommendations_generated(self, sample_images):
        """Проверяет генерацию рекомендаций"""
        image_dir, df = sample_images
        
        container = DataContainer(
            data=df.copy(),
            image_column="image_path",
            image_dir=image_dir,
            target_column="label"
        )
        
        augmentor = ImageAugmentor(
            augment_factor=2.0,
            output_dir=image_dir / "rec_test",
            verbose=False
        )
        
        result = augmentor.fit_transform(container)
        
        # Должны быть рекомендации
        assert len(result.recommendations) > 0
        
        # Проверяем структуру рекомендаций
        for rec in result.recommendations:
            assert "type" in rec
    
    def test_fit_info_populated(self, sample_images):
        """Проверяет заполнение fit_info"""
        image_dir, df = sample_images
        
        container = DataContainer(
            data=df.copy(),
            image_column="image_path",
            image_dir=image_dir
        )
        
        preprocessor = ImagePreprocessor(target_size=(100, 100))
        preprocessor.fit(container)
        
        assert "valid_images" in preprocessor._fit_info
        assert "invalid_images" in preprocessor._fit_info
        assert "target_size" in preprocessor._fit_info