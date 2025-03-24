from abc import ABC, abstractmethod
from typing import Any


class BaseSAE(ABC):
    @abstractmethod
    def load(self, *args, **kwargs) -> tuple[Any, str]:  # type: ignore
        pass
