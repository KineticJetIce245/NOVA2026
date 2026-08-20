from collections.abc import Callable
from typing import Any


class PipelineError(Exception):
    pass


class Pipeline[D]:
    def __init__(self):
        self.__tubes__: list[Callable[[D], tuple[Any, D]]] = []
        self.__load__: D | None = None
        self.__step_loc__: int = 0

    def add_tube(self, tube: Callable[[D], tuple[Any, D]]):
        self.__tubes__.append(tube)

    def rundown(self, data: D):
        temp_load = data
        result = None
        for tube in self.__tubes__:
            result, temp_load = tube(temp_load)
        return result, temp_load

    def feed(self, data: D):
        if self.__load__ is None:
            self.__load__ = data
            self.__step_loc__ = 0
        else:
            raise PipelineError(
                "Pipeline is already processing data. Call step() first."
            )

    def step(self, steps: int = 1):
        if len(self.__tubes__) == 0:
            raise PipelineError("Pipeline has no tubes to process data.")
        if steps <= 0:
            raise PipelineError("Number of steps must be positive.")
        if self.__load__ is None:
            raise PipelineError("Pipeline has no data to process. Call feed() first.")
        if self.__step_loc__ >= len(self.__tubes__):
            self.__load__ = None
            self.__step_loc__ = 0
            raise PipelineError(
                "Pipeline has already completed processing. Call feed() first."
            )

        result = None
        steps_forward = min(steps, len(self.__tubes__) - self.__step_loc__)

        for _ in range(steps_forward):
            result, self.__load__ = self.__tubes__[self.__step_loc__](self.__load__)
            self.__step_loc__ += 1

        load = self.__load__
        if self.__step_loc__ >= len(self.__tubes__):
            self.__load__ = None
            self.__step_loc__ = 0
        return result, load

    def spit(self):
        load = self.__load__
        self.__step_loc__ = 0
        self.__load__ = None
        return load
