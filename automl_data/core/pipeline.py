# automl_data/core/pipeline.py
"""
Pipeline — оркестратор обработки.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import time
import logging

from .container import DataContainer, ProcessingStep, ProcessingStage
from ..utils.exceptions import PipelineError


@dataclass
class PipelineResult:
    """Результат выполнения пайплайна"""
    container: DataContainer
    success: bool
    execution_time: float = 0.0
    steps_executed: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


class Pipeline:
    """
    Пайплайн обработки данных.
    
    Example:
        >>> pipeline = Pipeline()
        >>> pipeline.add_step(EncodingAdapter())
        >>> result = pipeline.execute(container)
    """
    
    def __init__(self, name: str = "Pipeline", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self._steps: list[dict[str, Any]] = []
        self._logger = logging.getLogger(f"automl_data.{name}")
    
    def add_step(
        self,
        component: Any,
        name: str | None = None,
        condition: Callable[[DataContainer], bool] | None = None,
        on_error: str = "warn"
    ) -> Pipeline:
        """Добавить шаг (fluent API)"""
        self._steps.append({
            "component": component,
            "name": name or component.__class__.__name__,
            "condition": condition,
            "on_error": on_error
        })
        return self
    
    def execute(self, container: DataContainer) -> PipelineResult:
        """Выполнить пайплайн"""
        start = time.time()
        
        result = PipelineResult(
            container=container.clone(),
            success=True
        )
        
        for step in self._steps:
            step_name = step["name"]
            component = step["component"]
            condition = step["condition"]
            on_error = step["on_error"]
            
            if condition and not condition(result.container):
                self._log(f"Skipping {step_name}")
                continue
            
            try:
                self._log(f"Executing: {step_name}")
                step_start = time.time()
                
                if hasattr(component, 'fit_transform'):
                    result.container = component.fit_transform(result.container)
                elif hasattr(component, 'transform'):
                    result.container = component.transform(result.container)
                elif callable(component):
                    result.container = component(result.container)
                
                duration = time.time() - step_start
                result.steps_executed.append(step_name)
                self._log(f"  Completed in {duration:.2f}s")
                
            except Exception as e:
                if on_error == "raise":
                    result.success = False
                    raise PipelineError(str(e), step=step_name, original_error=e)
                elif on_error == "warn":
                    self._log(f"  Warning: {e}", level="warning")
                    result.errors.append({"step": step_name, "error": str(e)})
        
        result.execution_time = time.time() - start
        result.container.stage = ProcessingStage.READY
        
        return result
    
    def _log(self, message: str, level: str = "info"):
        if self.verbose:
            getattr(self._logger, level)(message)
    
    def __len__(self) -> int:
        return len(self._steps)
    
    def __repr__(self) -> str:
        steps = [s["name"] for s in self._steps]
        return f"Pipeline(name='{self.name}', steps={steps})"