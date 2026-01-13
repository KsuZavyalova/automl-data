# automl_data/utils/dependencies.py
"""
Управление зависимостями и ленивой загрузкой.
Реализует Singleton для кэширования состояния окружения.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from functools import lru_cache

from .exceptions import DependencyError
from .decorators import singleton

@lru_cache(maxsize=64)
def _check_import(package: str) -> bool:
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False


def require_package(
    package: str, 
    install_name: str | None = None,
    feature: str | None = None
) -> None:
    """
    Проверка наличия пакета с кэшированием.
    """
    if not _check_import(package):
        raise DependencyError(package, install_name, feature)


def optional_import(package: str, submodule: str | None = None) -> Any | None:
    try:
        module = importlib.import_module(package)
        if submodule:
            return getattr(module, submodule, None)
        return module
    except ImportError:
        return None


@singleton
class DependencyManager:
    """
    Менеджер зависимостей (Singleton).
    Централизованно хранит информацию о всех используемых в проекте библиотеках.
    """
    
    # Полная карта зависимостей из всех адаптеров
    # Format: import_name -> (pip_install_name, category)
    PACKAGES = {
        # Core & Tabular
        "sklearn": ("scikit-learn", "Core"),
        "pandas": ("pandas", "Core"),
        "numpy": ("numpy", "Core"),
        "scipy": ("scipy", "Core"),
        "category_encoders": ("category-encoders", "Tabular"),
        "pyod": ("pyod", "Tabular"),
        "imblearn": ("imbalanced-learn", "Tabular"),
        "ydata_profiling": ("ydata-profiling", "Tabular"),
        
        # Text Processing
        "nltk": ("nltk", "Text"),
        "transformers": ("transformers", "Text"),
        "torch": ("torch", "Text/DL"),
        
        # Image Processing
        "cv2": ("opencv-python", "Image"),
        "albumentations": ("albumentations", "Image"),
        "torchvision": ("torchvision", "Image"),
        "PIL": ("Pillow", "Image"),
    }
    
    def __init__(self):
        self._status = {}
        self._versions = {}
    
    def check_all(self) -> dict[str, bool]:
        """Проверка статуса всех зависимостей (ленивая)"""
        if not self._status:
            for name in self.PACKAGES:
                is_installed = _check_import(name)
                self._status[name] = is_installed
                if is_installed:
                    self._versions[name] = get_version(name)
        return self._status.copy()
    
    def get_missing(self) -> list[tuple[str, str]]:
        """Возвращает список отсутствующих пакетов (import_name, install_name)"""
        status = self.check_all()
        missing = []
        for name, is_installed in status.items():
            if not is_installed:
                install_name = self.PACKAGES[name][0]
                missing.append((name, install_name))
        return missing

    def print_status(self) -> None:
        """Красивый отчет о состоянии окружения"""
        self.check_all()
        
        print("\n📦 AutoForge Environment Status\n")
        print(f"{'Package':<20} {'Status':<8} {'Version':<15} {'Category':<10}")
        print("-" * 60)
        
        # Сортируем по категориям
        sorted_pkgs = sorted(self.PACKAGES.items(), key=lambda x: (x[1][1], x[0]))
        
        current_cat = ""
        for pkg, (pip_name, cat) in sorted_pkgs:
            if cat != current_cat:
                print(f"\n--- {cat} ---")
                current_cat = cat
                
            is_installed = self._status.get(pkg, False)
            status_icon = "✅" if is_installed else "❌"
            version = self._versions.get(pkg, "-")
            
            print(f"{pkg:<20} {status_icon:<8} {version:<15}")
            
        missing = self.get_missing()
        if missing:
            print("\n" + "-" * 60)
            print("💡 Missing optional packages:")
            for pkg, install in missing:
                print(f"   pip install {install}")


def check_dependencies() -> dict[str, bool]:
    return DependencyManager().check_all()

def print_dependency_status() -> None:
    DependencyManager().print_status()

def get_version(package: str) -> str | None:
    try:
        module = importlib.import_module(package)
        return getattr(module, "__version__", "unknown")
    except ImportError:
        return None


class LazyImport:
    """Ленивый импорт модуля."""
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