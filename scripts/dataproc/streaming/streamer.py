from collections.abc import Callable
from typing import Any

from acquisition_queue import AcquisitionQueue
from streams import DefaultStreamer


class Streamer:
    def __init__(self) -> None:
        self.__pre_processing_tubes__: list[Callable[[Any], Any]] = []

    def add_pre_processing_tube(
        self, pre_processing_step: Callable[[Any], Any]
    ) -> None:
        self.__pre_processing_tubes__.append(pre_processing_step)
