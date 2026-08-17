from isanlp import PipelineCommon
from isanlp.ru.converter_mystem_to_ud import ConverterMystemToUd
from isanlp.ru.processor_mystem import ProcessorMystem

from .base_features_extractor import BaseFeaturesExtractor


class FeaturesMystem(BaseFeaturesExtractor):
    # vendoring patch: pipeline построен в __init__ (в оригинале — атрибут
    # класса), чтобы mystem-подпроцесс можно было пересоздать новым экземпляром
    def __init__(self):
        super().__init__()
        self.pipeline = PipelineCommon([
            (
                ProcessorMystem(),
                ['tokens', 'sentences'],
                {'postag': 'postag_mys_unconverted',
                 'lemma': 'lemma_mys'}),
            (
                ConverterMystemToUd(),
                ['postag_mys_unconverted'],
                {'morph': 'morph_mys',
                 'postag': 'postag_mys'})
        ])

    def __call__(self, tokens, sentences) -> dict:
        return self.pipeline(tokens, sentences)
