"""
Централизованное логирование для automl_data.

Обеспечивает:
- Единый формат логов
- Уровни логирования
- Разные обработчики (консоль, файл)
- Цветное форматирование (опционально)
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)


class ColorFormatter(logging.Formatter):
    """Форматирование логов с цветами"""
    
    COLOR_MAP = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT
    }
    
    def format(self, record):
        # Добавляем цвет в зависимости от уровня
        level_color = self.COLOR_MAP.get(record.levelno, Fore.WHITE)
        
        # Форматируем сообщение
        if self._style is not None:
            # Для Python 3.8+
            message = super().format(record)
        else:
            message = logging.Formatter.format(self, record)
        
        # Добавляем цвет к уровню логирования
        if '[' in message and ']' in message.split()[0]:
            # Находим уровень логирования и добавляем цвет
            parts = message.split(' ', 1)
            if len(parts) > 1:
                level_part = parts[0]
                rest = parts[1]
                colored_level = level_part.replace(
                    record.levelname,
                    f"{level_color}{record.levelname}{Style.RESET_ALL}"
                )
                message = colored_level + ' ' + rest
        
        return message


class AutoMLLogger:
    """
    Главный логгер для библиотеки automl_data.
    
    Usage:
        >>> from automl_data.utils.logging import get_logger
        >>> logger = get_logger("ModuleName")
        >>> logger.info("Message")
    """
    
    _loggers: Dict[str, 'AutoMLLogger'] = {}
    _configured = False
    
    def __init__(self, name: str = "automl_data"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        if not AutoMLLogger._configured:
            self._configure_default()
    
    @classmethod
    def get_logger(cls, name: str = "automl_data") -> 'AutoMLLogger':
        """Получить или создать логгер"""
        if name not in cls._loggers:
            cls._loggers[name] = AutoMLLogger(name)
        return cls._loggers[name]
    
    def _configure_default(self):
        """Конфигурация логгера по умолчанию"""
        if AutoMLLogger._configured:
            return
        
        # Форматтер
        fmt = '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s'
        datefmt = '%H:%M:%S'
        
        # Цветной форматтер для консоли
        console_formatter = ColorFormatter(fmt, datefmt)
        
        # Простой форматтер для файла
        file_formatter = logging.Formatter(fmt, datefmt)
        
        # Обработчик для консоли
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Добавляем обработчик к корневому логгеру
        root_logger = logging.getLogger("automl_data")
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.INFO)
        
        AutoMLLogger._configured = True
    
    def configure(
        self,
        level: int = logging.INFO,
        console: bool = True,
        file_path: Optional[Path] = None,
        file_level: int = logging.DEBUG,
        propagate: bool = False,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None
    ):
        """
        Конфигурация логгера.
        
        Args:
            level: Уровень логирования
            console: Выводить ли в консоль
            file_path: Путь к файлу логов
            file_level: Уровень логирования в файл
            propagate: Пропагировать ли сообщения в родительские логгеры
            fmt: Формат сообщения
            datefmt: Формат даты/времени
        """
        self.logger.setLevel(level)
        self.logger.propagate = propagate
        
        # Удаляем существующие обработчики
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Формат по умолчанию
        if fmt is None:
            fmt = '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s'
        if datefmt is None:
            datefmt = '%H:%M:%S'
        
        # Консольный обработчик
        if console:
            console_formatter = ColorFormatter(fmt, datefmt)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Файловый обработчик
        if file_path:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_formatter = logging.Formatter(fmt, datefmt)
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, msg: str, *args, **kwargs):
        """Логирование DEBUG уровня"""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Логирование INFO уровня"""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Логирование WARNING уровня"""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Логирование ERROR уровня"""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Логирование CRITICAL уровня"""
        self.logger.critical(msg, *args, **kwargs)
    
    def log(self, level: int, msg: str, *args, **kwargs):
        """Логирование произвольного уровня"""
        self.logger.log(level, msg, *args, **kwargs)
    
    def progress(self, msg: str, current: int, total: int):
        """Логирование прогресса"""
        percent = current / total * 100 if total > 0 else 0
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        progress_msg = f"{msg} [{bar}] {percent:.1f}% ({current}/{total})"
        self.info(progress_msg)
    
    def section(self, title: str, char: str = "="):
        """Логирование раздела"""
        line = char * 50
        self.info(f"\n{line}")
        self.info(f"{title.center(50)}")
        self.info(f"{line}\n")
    
    def success(self, msg: str):
        """Логирование успешного завершения"""
        self.info(f"{Fore.GREEN}✓{Style.RESET_ALL} {msg}")
    
    def failure(self, msg: str):
        """Логирование ошибки"""
        self.error(f"{Fore.RED}✗{Style.RESET_ALL} {msg}")
    
    def warning_icon(self, msg: str):
        """Логирование предупреждения с иконкой"""
        self.warning(f"{Fore.YELLOW}⚠{Style.RESET_ALL} {msg}")
    
    def set_level(self, level: int):
        """Установить уровень логирования"""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
    
    def add_handler(self, handler: logging.Handler):
        """Добавить обработчик логов"""
        self.logger.addHandler(handler)
    
    def get_child(self, suffix: str) -> 'AutoMLLogger':
        """Получить дочерний логгер"""
        child_name = f"{self.name}.{suffix}"
        return AutoMLLogger.get_logger(child_name)


def get_logger(name: str = "automl_data") -> AutoMLLogger:
    """Получить логгер"""
    return AutoMLLogger.get_logger(name)


def setup_logging(
    level: int = logging.INFO,
    console: bool = True,
    file_path: Optional[Path] = None,
    propagate: bool = False
):
    """
    Настройка логирования для всей библиотеки.
    
    Args:
        level: Уровень логирования
        console: Выводить ли в консоль
        file_path: Путь к файлу логов
        propagate: Пропагировать ли сообщения
    """
    logger = get_logger()
    logger.configure(
        level=level,
        console=console,
        file_path=file_path,
        propagate=propagate
    )


# Пример использования декораторов с логированием
def log_timing(func):
    """Декоратор для логирования времени выполнения"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.perf_counter()
        
        logger.debug(f"Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            logger.debug(f"Finished {func.__name__} in {elapsed:.4f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.4f}s: {e}")
            raise
    
    return wrapper


def log_call(level=logging.DEBUG):
    """Декоратор для логирования вызовов функций"""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            # Формируем строку аргументов
            args_str = ', '.join(map(str, args))
            kwargs_str = ', '.join(f"{k}={v}" for k, v in kwargs.items())
            all_args = ', '.join(filter(None, [args_str, kwargs_str]))
            
            logger.log(level, f"Calling {func.__name__}({all_args})")
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func.__name__} returned: {result}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised exception: {e}")
                raise
        
        return wrapper
    return decorator