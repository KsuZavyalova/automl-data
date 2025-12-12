# automl_data/utils/dependencies.py
"""
Управление зависимостями.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from .exceptions import DependencyError


def require_package(
    package: str, 
    install_name: str | None = None,
    feature: str | None = None
) -> None:
    """
    Проверка наличия пакета. Выбрасывает исключение если не установлен.
    
    Args:
        package: Имя пакета для импорта
        install_name: Имя для pip install (если отличается)
        feature: Название функции, для которой нужен пакет
    
    Raises:
        DependencyError: Если пакет не установлен
    
    Example:
        >>> require_package("sklearn", "scikit-learn", "preprocessing")
    """
    try:
        importlib.import_module(package)
    except ImportError:
        raise DependencyError(package, install_name, feature)


def optional_import(
    package: str, 
    submodule: str | None = None
) -> Any | None:
    """
    Опциональный импорт пакета.
    
    Args:
        package: Имя пакета
        submodule: Подмодуль для импорта
    
    Returns:
        Модуль или None если не установлен
    
    Example:
        >>> cv2 = optional_import("cv2")
        >>> if cv2:
        ...     img = cv2.imread("image.jpg")
    """
    try:
        module = importlib.import_module(package)
        if submodule:
            return getattr(module, submodule, None)
        return module
    except ImportError:
        return None


def check_dependencies() -> dict[str, bool]:
    """
    Проверка всех опциональных зависимостей.
    
    Returns:
        Словарь {имя_пакета: установлен}
    
    Example:
        >>> deps = check_dependencies()
        >>> print(deps)
        {'sklearn': True, 'nlpaug': False, ...}
    """
    packages = {
        # Табличные данные
        "sklearn": "scikit-learn",
        "category_encoders": "category-encoders",
        "pyod": "pyod",
        "imblearn": "imbalanced-learn",
        "ydata_profiling": "ydata-profiling",
        
        # Текст
        "nltk": "nltk",
        "transformers": "transformers",
        
        # Изображения
        "albumentations": "albumentations",
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "torchvision": "torchvision",
    }
    
    return {
        name: optional_import(name) is not None
        for name in packages
    }


def print_dependency_status() -> None:
    """Красивый вывод статуса зависимостей"""
    deps = check_dependencies()
    
    print("\n📦 ML Data Forge - Dependency Status\n")
    print("-" * 45)
    
    categories = {
        "Tabular Data": ["sklearn", "category_encoders", "pyod", "imblearn", "ydata_profiling"],
        "Text Data": ["nlpaug", "nltk", "transformers"],
        "Image Data": ["albumentations", "cv2", "PIL", "torchvision"],
    }
    
    for category, packages in categories.items():
        print(f"\n{category}:")
        for pkg in packages:
            status = "✅" if deps.get(pkg, False) else "❌"
            print(f"  {status} {pkg}")
    
    print("\n" + "-" * 45)
    
    missing = [k for k, v in deps.items() if not v]
    if missing:
        print(f"\n💡 To install missing packages:")
        print(f"   pip install automl-data[full]")


def get_version(package: str) -> str | None:
    """Получить версию установленного пакета"""
    try:
        module = importlib.import_module(package)
        return getattr(module, "__version__", "unknown")
    except ImportError:
        return None


class LazyImport:
    """
    Ленивый импорт модуля.
    
    Модуль импортируется только при первом обращении к нему.
    
    Example:
        >>> np = LazyImport("numpy")
        >>> # numpy ещё не импортирован
        >>> arr = np.array([1, 2, 3])  # Теперь импортирован
    """
    
    def __init__(self, module_name: str, install_name: str | None = None):
        self._module_name = module_name
        self._install_name = install_name
        self._module = None
    
    def _load(self):
        if self._module is None:
            require_package(self._module_name, self._install_name)
            self._module = importlib.import_module(self._module_name)
        return self._module
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)
    
    def __repr__(self) -> str:
        loaded = "loaded" if self._module else "not loaded"
        return f"LazyImport('{self._module_name}', {loaded})"