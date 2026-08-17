from isanlp import PipelineCommon
from isanlp.processor_razdel import ProcessorRazdel

from .base_features_extractor import BaseFeaturesExtractor


class FeaturesRazdel(BaseFeaturesExtractor):
    # vendoring patch: pipeline построен в __init__ (в оригинале — атрибут
    # класса), чтобы экземпляры были независимы и пересоздаваемы
    def __init__(self):
        super().__init__()
        self.pipeline = PipelineCommon([
            (ProcessorRazdel(), ['text'],
             {'tokens': 'tokens',
              'sentences': 'sentences'}),
        ])

    def __call__(self, text: str) -> dict:
        return self.pipeline(text)
