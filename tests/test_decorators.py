# tests/test_decorators.py
"""
Юнит-тесты для декораторов.

Покрывает:
- @timing - измерение времени функций
- @timing_method - измерение времени методов
- @require_fitted - проверка обученности
- @safe_transform - безопасная трансформация
- @preserve_target - сохранение target
- @retry - повторные попытки
- @singleton - паттерн Singleton
- CountCalls - подсчёт вызовов
"""

from __future__ import annotations

import time
import pytest
import pandas as pd
import numpy as np

from automl_data.utils.decorators import (
    timing,
    timing_method,
    require_fitted,
    safe_transform,
    preserve_target,
    sync_container,
    retry,
    singleton,
    CountCalls
)
from automl_data.utils.exceptions import NotFittedError
from automl_data.core.container import DataContainer


# ==================== TIMING TESTS ====================

class TestTimingDecorator:
    """Тесты для @timing"""
    
    def test_measures_execution_time(self):
        """Проверяет измерение времени"""
        @timing
        def slow_function():
            time.sleep(0.05)
            return "done"
        
        result = slow_function()
        
        assert result == "done"
        assert hasattr(slow_function, 'last_execution_time')
        assert slow_function.last_execution_time >= 0.05
        assert slow_function.last_execution_time < 0.2
    
    def test_preserves_function_name(self):
        """Проверяет сохранение имени функции"""
        @timing
        def my_named_function():
            """Docstring"""
            pass
        
        assert my_named_function.__name__ == "my_named_function"
        assert my_named_function.__doc__ == "Docstring"
    
    def test_works_with_arguments(self):
        """Проверяет работу с аргументами"""
        @timing
        def add(a, b, *, multiplier=1):
            return (a + b) * multiplier
        
        result = add(2, 3, multiplier=2)
        
        assert result == 10
        assert add.last_execution_time >= 0
    
    def test_initial_time_is_zero(self):
        """Проверяет начальное значение"""
        @timing
        def func():
            pass
        
        assert func.last_execution_time == 0.0
    
    def test_time_updates_on_each_call(self):
        """Проверяет обновление времени"""
        @timing
        def variable_sleep(duration):
            time.sleep(duration)
        
        variable_sleep(0.02)
        time1 = variable_sleep.last_execution_time
        
        variable_sleep(0.05)
        time2 = variable_sleep.last_execution_time
        
        assert time2 > time1


class TestTimingMethodDecorator:
    """Тесты для @timing_method"""
    
    def test_stores_time_in_instance(self):
        """Проверяет сохранение времени в экземпляре"""
        class MyClass:
            _last_duration = 0.0
            
            @timing_method('_last_duration')
            def process(self):
                time.sleep(0.02)
                return "processed"
        
        obj = MyClass()
        result = obj.process()
        
        assert result == "processed"
        assert obj._last_duration >= 0.02
    
    def test_different_instances_independent(self):
        """Проверяет независимость экземпляров"""
        class MyClass:
            _duration = 0.0
            
            @timing_method('_duration')
            def work(self, sleep_time):
                time.sleep(sleep_time)
        
        obj1 = MyClass()
        obj2 = MyClass()
        
        obj1.work(0.02)
        obj2.work(0.05)
        
        assert obj1._duration < obj2._duration
    
    def test_custom_attribute_name(self):
        """Проверяет кастомное имя атрибута"""
        class MyClass:
            execution_time = 0.0
            
            @timing_method('execution_time')
            def run(self):
                time.sleep(0.01)
        
        obj = MyClass()
        obj.run()
        
        assert obj.execution_time >= 0.01


# ==================== REQUIRE_FITTED TESTS ====================

class TestRequireFittedDecorator:
    """Тесты для @require_fitted"""
    
    def test_raises_when_not_fitted(self):
        """Проверяет исключение для необученного объекта"""
        class Model:
            _is_fitted = False
            
            @require_fitted
            def predict(self, x):
                return x * 2
        
        model = Model()
        
        with pytest.raises(NotFittedError) as exc_info:
            model.predict(5)
        
        assert "not fitted" in str(exc_info.value).lower()
    
    def test_works_when_fitted(self):
        """Проверяет работу для обученного объекта"""
        class Model:
            _is_fitted = True
            
            @require_fitted
            def predict(self, x):
                return x * 2
        
        model = Model()
        result = model.predict(5)
        
        assert result == 10
    
    def test_dynamic_fitting(self):
        """Проверяет динамическое изменение состояния"""
        class Model:
            _is_fitted = False
            
            def fit(self):
                self._is_fitted = True
                return self
            
            @require_fitted
            def predict(self, x):
                return x * 2
        
        model = Model()
        
        with pytest.raises(NotFittedError):
            model.predict(5)
        
        model.fit()
        result = model.predict(5)
        
        assert result == 10
    
    def test_uses_name_attribute(self):
        """Проверяет использование _name в сообщении"""
        class Model:
            _is_fitted = False
            _name = "CustomModel"
            
            @require_fitted
            def predict(self, x):
                return x
        
        model = Model()
        
        with pytest.raises(NotFittedError) as exc_info:
            model.predict(5)
        
        assert "CustomModel" in str(exc_info.value)
    
    def test_missing_is_fitted_attribute(self):
        """Проверяет поведение при отсутствии _is_fitted"""
        class BadModel:
            @require_fitted
            def predict(self, x):
                return x
        
        model = BadModel()
        
        with pytest.raises(NotFittedError):
            model.predict(5)


# ==================== SAFE_TRANSFORM TESTS ====================

class TestSafeTransformDecorator:
    """Тесты для @safe_transform"""
    
    @pytest.fixture
    def sample_container(self):
        """Создаёт тестовый контейнер"""
        df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        return DataContainer(data=df, target_column='target')
    
    def test_preserves_target(self, sample_container):
        """Проверяет сохранение target"""
        class Adapter:
            @safe_transform(preserve_target=True)
            def transform(self, container):
                # Пытаемся изменить target
                container.data['target'] = container.data['target'] * 10
                return container
        
        original_target = sample_container.data['target'].copy()
        adapter = Adapter()
        
        result = adapter.transform(sample_container)
        
        # Target должен сохраниться
        np.testing.assert_array_equal(
            result.data['target'].values,
            original_target.values
        )
    
    def test_reset_index(self, sample_container):
        """Проверяет сброс индекса"""
        class Adapter:
            @safe_transform(reset_index=True)
            def transform(self, container):
                # Удаляем строки — индекс станет непоследовательным
                container.data = container.data.iloc[[0, 2, 4]]
                return container
        
        adapter = Adapter()
        result = adapter.transform(sample_container)
        
        # Индекс должен быть сброшен
        assert list(result.data.index) == [0, 1, 2]
    
    def test_sync_state_called(self, sample_container):
        """Проверяет вызов _sync_internal_state"""
        sync_called = [False]
        
        class MockContainer:
            def __init__(self, data):
                self.data = data
                self.target_column = 'target'
            
            def _sync_internal_state(self):
                sync_called[0] = True
        
        class Adapter:
            @safe_transform(sync_state=True)
            def transform(self, container):
                return container
        
        mock = MockContainer(sample_container.data.copy())
        adapter = Adapter()
        adapter.transform(mock)
        
        assert sync_called[0] is True
    
    def test_no_sync_when_disabled(self, sample_container):
        """Проверяет отключение синхронизации"""
        sync_called = [False]
        
        class MockContainer:
            def __init__(self, data):
                self.data = data
                self.target_column = 'target'
            
            def _sync_internal_state(self):
                sync_called[0] = True
        
        class Adapter:
            @safe_transform(sync_state=False)
            def transform(self, container):
                return container
        
        mock = MockContainer(sample_container.data.copy())
        adapter = Adapter()
        adapter.transform(mock)
        
        assert sync_called[0] is False


# ==================== PRESERVE_TARGET TESTS ====================

class TestPreserveTargetDecorator:
    """Тесты для @preserve_target"""
    
    def test_restores_modified_target(self):
        """Проверяет восстановление изменённого target"""
        df = pd.DataFrame({
            'feature': [1, 2, 3],
            'target': [10, 20, 30]
        })
        container = DataContainer(data=df, target_column='target')
        original_target = container.data['target'].copy()
        
        class Adapter:
            @preserve_target
            def transform(self, container):
                container.data['target'] = 0  # Меняем target
                return container
        
        adapter = Adapter()
        result = adapter.transform(container)
        
        np.testing.assert_array_equal(
            result.data['target'].values,
            original_target.values
        )
    
    def test_works_without_target(self):
        """Проверяет работу без target"""
        df = pd.DataFrame({'feature': [1, 2, 3]})
        container = DataContainer(data=df)
        
        class Adapter:
            @preserve_target
            def transform(self, container):
                container.data['feature'] = container.data['feature'] * 2
                return container
        
        adapter = Adapter()
        result = adapter.transform(container)
        
        assert list(result.data['feature']) == [2, 4, 6]


# ==================== RETRY TESTS ====================

class TestRetryDecorator:
    """Тесты для @retry"""
    
    def test_succeeds_first_try(self):
        """Проверяет успех с первой попытки"""
        call_count = [0]
        
        @retry(max_attempts=3, delay=0.01)
        def always_works():
            call_count[0] += 1
            return "success"
        
        result = always_works()
        
        assert result == "success"
        assert call_count[0] == 1
    
    def test_retries_on_failure(self):
        """Проверяет повторные попытки"""
        attempts = []
        
        @retry(max_attempts=3, delay=0.01)
        def fails_twice():
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = fails_twice()
        
        assert result == "success"
        assert len(attempts) == 3
    
    def test_raises_after_max_attempts(self):
        """Проверяет исключение после исчерпания попыток"""
        @retry(max_attempts=3, delay=0.01)
        def always_fails():
            raise RuntimeError("Always fails")
        
        with pytest.raises(RuntimeError) as exc_info:
            always_fails()
        
        assert "Always fails" in str(exc_info.value)
    
    def test_only_catches_specified_exceptions(self):
        """Проверяет перехват только указанных исключений"""
        @retry(max_attempts=3, delay=0.01, exceptions=(ValueError,))
        def raises_type_error():
            raise TypeError("Wrong type")
        
        with pytest.raises(TypeError):
            raises_type_error()
    
    def test_on_retry_callback(self):
        """Проверяет callback при повторе"""
        retry_log = []
        
        def log_retry(exc, attempt):
            retry_log.append((str(exc), attempt))
        
        attempts = [0]
        
        @retry(max_attempts=3, delay=0.01, on_retry=log_retry)
        def fails_then_succeeds():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError(f"Attempt {attempts[0]}")
            return "done"
        
        result = fails_then_succeeds()
        
        assert result == "done"
        assert len(retry_log) == 2
    
    def test_exponential_backoff(self):
        """Проверяет экспоненциальную задержку"""
        times = []
        
        @retry(max_attempts=3, delay=0.02, backoff=2.0)
        def track_time():
            times.append(time.perf_counter())
            if len(times) < 3:
                raise ValueError("retry")
            return "done"
        
        track_time()
        
        delay1 = times[1] - times[0]
        delay2 = times[2] - times[1]
        
        # Вторая задержка должна быть больше
        assert delay2 > delay1


# ==================== SINGLETON TESTS ====================

class TestSingletonDecorator:
    """Тесты для @singleton"""
    
    def test_same_instance(self):
        """Проверяет один и тот же экземпляр"""
        @singleton
        class Database:
            def __init__(self):
                self.id = id(self)
        
        db1 = Database()
        db2 = Database()
        
        assert db1 is db2
        assert db1.id == db2.id
    
    def test_init_called_once(self):
        """Проверяет однократный вызов __init__"""
        init_count = [0]
        
        @singleton
        class Counter:
            def __init__(self):
                init_count[0] += 1
        
        Counter()
        Counter()
        Counter()
        
        assert init_count[0] == 1
    
    def test_preserves_class_name(self):
        """Проверяет сохранение имени класса"""
        @singleton
        class MyService:
            """Service docstring"""
            pass
        
        assert MyService.__name__ == "MyService"
        assert MyService.__doc__ == "Service docstring"
    
    def test_different_singletons_independent(self):
        """Проверяет независимость разных синглтонов"""
        @singleton
        class ServiceA:
            value = "A"
        
        @singleton
        class ServiceB:
            value = "B"
        
        a = ServiceA()
        b = ServiceB()
        
        assert a is not b
        assert a.value == "A"
        assert b.value == "B"


# ==================== COUNT_CALLS TESTS ====================

class TestCountCallsDecorator:
    """Тесты для CountCalls"""
    
    def test_counts_calls(self):
        """Проверяет подсчёт вызовов"""
        @CountCalls
        def my_func():
            return "result"
        
        assert my_func.call_count == 0
        
        my_func()
        assert my_func.call_count == 1
        
        my_func()
        my_func()
        assert my_func.call_count == 3
    
    def test_returns_correct_result(self):
        """Проверяет возврат результата"""
        @CountCalls
        def add(a, b):
            return a + b
        
        result = add(2, 3)
        
        assert result == 5
        assert add.call_count == 1
    
    def test_reset(self):
        """Проверяет сброс счётчика"""
        @CountCalls
        def my_func():
            pass
        
        my_func()
        my_func()
        assert my_func.call_count == 2
        
        my_func.reset()
        assert my_func.call_count == 0
    
    def test_preserves_metadata(self):
        """Проверяет сохранение метаданных"""
        @CountCalls
        def documented():
            """My docstring"""
            pass
        
        assert documented.__name__ == "documented"
        assert documented.__doc__ == "My docstring"
    
    def test_with_exception(self):
        """Проверяет подсчёт при исключениях"""
        @CountCalls
        def may_fail(fail):
            if fail:
                raise ValueError("Failed")
            return "ok"
        
        may_fail(False)
        assert may_fail.call_count == 1
        
        with pytest.raises(ValueError):
            may_fail(True)
        
        assert may_fail.call_count == 2


# ==================== INTEGRATION TESTS ====================

class TestDecoratorsCombinations:
    """Тесты комбинаций декораторов"""
    
    def test_timing_with_retry(self):
        """Проверяет @timing + @retry"""
        attempts = [0]
        
        @timing
        @retry(max_attempts=3, delay=0.01)
        def flaky_function():
            attempts[0] += 1
            if attempts[0] < 2:
                raise ValueError("First fails")
            time.sleep(0.02)
            return "success"
        
        result = flaky_function()
        
        assert result == "success"
        assert flaky_function.last_execution_time >= 0.02
 