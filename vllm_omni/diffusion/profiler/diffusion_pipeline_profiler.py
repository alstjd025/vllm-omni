import functools
import os
import re
import time
from collections.abc import Callable
from threading import Lock
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


# [SparkGen] 환경 변수 SPARKGEN_STEP_TIMING=1이면 DiT(transformer, transformer_2) forward 호출 하나하나의 GPU 시간을
# stage_durations에 "<pipeline>.<module>.forward#<호출 순번>" key로 더한다. step마다 synchronize하면 CPU가 다음 step의
# kernel을 미리 넣어 두는 겹침이 사라져 측정 대상이 바뀌므로, 호출마다 CUDA event 두 개만 기록하고 요청(forward)이 끝나
# 이미 synchronize한 뒤에 한꺼번에 elapsed_time을 읽는다. CFG를 같은 GPU에서 하면 step 하나에 호출이 두 번 생긴다.
_STEP_TIMING_TARGETS = ("transformer.forward", "transformer_2.forward")


def _step_timing_enabled() -> bool:
    return os.environ.get("SPARKGEN_STEP_TIMING", "0") == "1"


def step_event_recorder(name: str, func: Callable, instance: Any) -> Callable:
    """DiT forward 한 번의 시작·끝에 CUDA event를 기록한다. 시간은 _resolve_step_events가 계산한다."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # pipeline forward 밖의 호출(초기화 때의 dummy 실행, step 단위 실행 모드)은 기록하지 않는다. 기록하면 resolve되지 않고 쌓인다
        if not instance._in_forward:
            return func(*args, **kwargs)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            return func(*args, **kwargs)
        finally:
            end.record()
            instance._step_events.append((name, start, end))

    return wrapper


def _resolve_step_events(instance: Any) -> None:
    """기록된 event 쌍을 초 단위 시간으로 바꿔 stage_durations에 넣는다. _profiler_lock을 잡은 상태에서 부른다.

    pipeline은 forward 안에서 stage_durations를 읽어 출력을 만든다(바깥 wrapper가 끝나기 전). 그래서 읽는 시점에 바꾼다.
    마지막 event만 기다리면 같은 stream에서 앞서 기록된 event는 모두 끝나 있다.
    """
    events, instance._step_events = instance._step_events, []
    if events:
        events[-1][2].synchronize()
    for i, (name, start, end) in enumerate(events):
        instance._stage_durations[f"{name}#{i:03d}"] = start.elapsed_time(end) / 1000.0


def profiler(name: str, func: Callable, instance: Any) -> Callable:
    """Timing a function execution."""
    metric_name = f"{name.rsplit('.', 1)[0]}.diffuse" if name.endswith(".denoise_step") else name
    is_forward = name == f"{instance.__class__.__name__}.forward"

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if is_forward:
            instance.clear_profiler_records()
            instance._in_forward = True
        if current_omni_platform.is_available():
            current_omni_platform.synchronize()
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            if current_omni_platform.is_available():
                current_omni_platform.synchronize()
            if is_forward:
                instance._in_forward = False
            duration = time.perf_counter() - start_time
            logger.info(f"[DiffusionPipelineProfiler] {metric_name} took {duration:.6f}s")
            # record the profiling data: duration of stages
            with instance._profiler_lock:
                instance._stage_durations[metric_name] = instance._stage_durations.get(metric_name, 0.0) + duration

    return wrapper


def _parse_part(part: str) -> tuple[str, int | None]:
    """Parse 'att[num]' into ('att', num)."""
    if m := re.compile(r"(\w+)\[(\d+)\]").fullmatch(part):
        return m.group(1), int(m.group(2))
    return part, None


def _get_attribute_by_path(obj: Any, path: str) -> tuple[Any, str]:
    """Traverse an object by dotted path and return (parent_obj, attribute_name)."""
    parts = path.split(".")
    current = obj

    for part in parts[:-1]:
        attr, idx = _parse_part(part)

        current = getattr(current, attr, None)
        if current is None:
            return None, None
        if idx is not None:
            current = current[idx]

    return current, parts[-1]


def wrap_methods_by_paths(root_obj: Any, method_paths: list[str]) -> None:
    """Wrap specified methods of an object with profiler."""
    for path in method_paths:
        obj, method_name = _get_attribute_by_path(root_obj, path)
        if not obj or not hasattr(obj, method_name):
            logger.warning(f"[DiffusionPipelineProfiler] Method path {path} not found")
            continue

        original_method = getattr(obj, method_name)
        if not callable(original_method):
            logger.warning(f"[DiffusionPipelineProfiler] Attribute {path} is not callable")
            continue

        profiler_name = f"{root_obj.__class__.__name__}.{path}"
        setattr(obj, method_name, profiler(profiler_name, original_method, root_obj))


class DiffusionPipelineProfilerMixin:
    _PROFILER_TARGETS = ["vae.encode", "vae.decode", "diffuse", "text_encoder.forward", "tokenizer.forward"]

    def setup_diffusion_pipeline_profiler(
        self,
        profiler_targets: list[str] | None = None,
        enable_diffusion_pipeline_profiler: bool = False,
    ) -> None:
        self.enable_diffusion_pipeline_profiler = enable_diffusion_pipeline_profiler
        if not enable_diffusion_pipeline_profiler:
            self.enable_diffusion_pipeline_profiler = enable_diffusion_pipeline_profiler
            return
        self._profiler_lock = Lock()
        self._stage_durations: dict[str, float] = {}
        targets = profiler_targets if profiler_targets is not None else self._PROFILER_TARGETS
        if not targets:
            targets = []
        else:
            targets = list(targets)
        if not profiler_targets and hasattr(self, "denoise_step"):
            targets.append("denoise_step")

        targets = ["forward"] + [
            t for t in targets if t != "forward"
        ]  # ensure "forward" implement 'clear_profiler_records' at first place

        targets = list(dict.fromkeys(targets))
        wrap_methods_by_paths(
            self,
            targets,
        )

        self._step_events: list = []
        self._in_forward = False
        if _step_timing_enabled() and torch.cuda.is_available():
            for path in _STEP_TIMING_TARGETS:
                obj, method_name = _get_attribute_by_path(self, path)
                if obj is None or not callable(getattr(obj, method_name, None)):
                    continue
                setattr(obj, method_name, step_event_recorder(path, getattr(obj, method_name), self))
                logger.info(f"[DiffusionPipelineProfiler] SPARKGEN_STEP_TIMING: recording {path} per call")

    @property
    def stage_durations(self) -> dict[str, float]:
        with self._profiler_lock:
            if getattr(self, "_step_events", None):
                _resolve_step_events(self)
            return self._stage_durations.copy()

    def clear_profiler_records(self) -> None:
        with self._profiler_lock:
            self._stage_durations.clear()
            self._step_events = []
