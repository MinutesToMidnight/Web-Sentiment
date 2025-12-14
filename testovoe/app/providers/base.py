# app/providers/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseProvider(ABC):
    @abstractmethod
    async def analyze(self, text: str, mode: str) -> Dict[str, Any]:
        """
        Выполняет анализ текста и возвращает "сырые" данные провайдера.
        Формат возвращаемого словаря зависит от реализации провайдера,
        но ожидается, что там будут ключи 'labels' (list of {label, score}),
        'summary' (str) и 'tone' (str) или совместимые аналоги.
        """
        raise NotImplementedError
