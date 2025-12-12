# tests/test_decorators.py
"""
Тесты для декораторов.
"""

import pytest
import time

from automl_data.utils.decorators import (
    timing, 
    require_fitted, 
    retry, 
    deprecated,
    validate_types,
    memoize,
    CountCalls
)
from automl_data.utils.exceptions import NotFittedError


class TestTimingDecorator:
    """Тесты @timing"""
    
    def test_measures_time(self):
        @timing
        def slow_function():
            time.sleep(0.1)
            return "done"
        
        result = slow_function()
        
        assert result == "done"
        assert slow_function.last_execution_time >= 0.1
    
    def test_preserves_return_value(self):
        @timing
        def add(a, b):
            return a + b
        
        assert add(2, 3) == 5


class TestRequireFittedDecorator:
    """Тесты @require_fitted"""
    
    def test_raises_when_not_fitted(self):
        class Model:
            _is_fitted = False
            
            @require_fitted
            def predict(self, x):
                return x * 2
        
        model = Model()
        
        with pytest.raises(NotFittedError):
            model.predict(5)
    
    def test_works_when_fitted(self):
        class Model:
            _is_fitted = True
            
            @require_fitted
            def predict(self, x):
                return x * 2
        
        model = Model()
        assert model.predict(5) == 10


class TestRetryDecorator:
    """Тесты @retry"""
    
    def test_retries_on_failure(self):
        call_count = 0
        
        @retry(max_attempts=3, delay=0.01)
        def unstable():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = unstable()
        
        assert result == "success"
        assert call_count == 3
    
    def test_raises_after_max_attempts(self):
        @retry(max_attempts=2, delay=0.01)
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fails()


class TestMemoizeDecorator:
    """Тесты @memoize"""
    
    def test_caches_result(self):
        call_count = 0
        
        @memoize
        def expensive(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # Первый вызов
        assert expensive(5) == 10
        assert call_count == 1
        
        # Второй вызов — из кэша
        assert expensive(5) == 10
        assert call_count == 1
        
        # Новый аргумент
        assert expensive(10) == 20
        assert call_count == 2


class TestCountCallsDecorator:
    """Тесты CountCalls"""
    
    def test_counts_calls(self):
        @CountCalls
        def my_func():
            return "hello"
        
        assert my_func.call_count == 0
        
        my_func()
        assert my_func.call_count == 1
        
        my_func()
        my_func()
        assert my_func.call_count == 3
    
    def test_reset(self):
        @CountCalls
        def my_func():
            pass
        
        my_func()
        my_func()
        my_func.reset()
        
        assert my_func.call_count == 0