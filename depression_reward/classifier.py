import json
from pathlib import Path

import numpy as np

SELF_CHECK_TOL = 1e-6
# эталонные decision_function на векторах из test_features.json
# (сверено с оригинальным Docker-контейнером, расхождение ~1e-16)
REFERENCE_SCORES = {'depr': -0.0557977018, 'norm': -0.1023075880}


def default_model_dir() -> Path:
    return Path(__file__).parent / 'model'


def build_model_from_weights(model_dir: Path):
    """Пересборка пайплайна из weights.npz + params.json под текущий sklearn.

    Логика идентична model/rebuild_model.py, но возвращает модель в памяти.
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    w = np.load(model_dir / 'weights.npz')
    p = json.loads((model_dir / 'params.json').read_text())

    arr = lambda k, dt=np.float64: np.ascontiguousarray(w[k], dtype=dt)

    scaler = StandardScaler(with_mean=p['scaler']['with_mean'], with_std=p['scaler']['with_std'])
    scaler.mean_ = arr('scaler_mean')
    scaler.scale_ = arr('scaler_scale')
    scaler.var_ = arr('scaler_var')
    scaler.n_samples_seen_ = p['scaler']['n_samples_seen']
    scaler.n_features_in_ = scaler.mean_.shape[0]

    pca = PCA(n_components=p['pca']['n_components'], whiten=p['pca']['whiten'],
              svd_solver=p['pca']['svd_solver'])
    pca.components_ = arr('pca_components')
    pca.mean_ = arr('pca_mean')
    pca.explained_variance_ = arr('pca_explained_variance')
    pca.explained_variance_ratio_ = arr('pca_explained_variance_ratio')
    pca.singular_values_ = arr('pca_singular_values')
    pca.noise_variance_ = p['pca']['noise_variance']
    pca.n_components_ = p['pca']['n_components']
    pca.n_features_in_ = pca.mean_.shape[0]
    pca.n_samples_ = p['svc']['shape_fit'][0]

    s = p['svc']
    svc = SVC(C=s['C'], kernel=s['kernel'], degree=s['degree'], gamma=s['gamma_computed'],
              coef0=s['coef0'], tol=s['tol'], shrinking=s['shrinking'],
              break_ties=s['break_ties'], decision_function_shape=s['decision_function_shape'],
              class_weight=s['class_weight_param'])
    svc.support_ = arr('svc_support', np.int32)
    svc.support_vectors_ = arr('svc_support_vectors')
    svc._n_support = arr('svc_n_support', np.int32)
    svc._dual_coef_ = arr('svc_dual_coef')
    svc._intercept_ = arr('svc_intercept')
    svc._probA = arr('svc_probA')
    svc._probB = arr('svc_probB')
    svc.classes_ = w['svc_classes']
    svc.class_weight_ = arr('svc_class_weight')
    svc._gamma = s['gamma_computed']
    svc._sparse = s['sparse']
    svc.shape_fit_ = tuple(s['shape_fit'])
    svc.fit_status_ = s['fit_status']
    svc.n_features_in_ = svc.support_vectors_.shape[1]
    # публичные dual_coef_/intercept_ — то же с флипом знака для бинарного c_svc
    svc.dual_coef_ = -svc._dual_coef_
    svc.intercept_ = -svc._intercept_

    return Pipeline([('rescaling', scaler), ('decomposition', pca), ('classifier', svc)])


def self_check(model, model_dir: Path, tol: float = SELF_CHECK_TOL) -> None:
    """Сверка decision_function с эталоном; RuntimeError при расхождении."""
    feats = json.loads((model_dir / 'test_features.json').read_text())
    for name, expected in REFERENCE_SCORES.items():
        x = np.array(feats[name], dtype=np.float64).reshape(1, -1)
        got = float(model.decision_function(x)[0])
        if abs(got - expected) > tol:
            raise RuntimeError(
                f'model self-check failed: {name} -> {got:.10f}, expected {expected:.10f}')


class StyleScorer:
    """Сборка классификатора из исходных весов и обязательная сверка.

    decision_function: выше = «депрессивнее». predict_proba у модели нет.
    """

    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self.model = self._load()

    def _load(self):
        model = build_model_from_weights(self.model_dir)
        self_check(model, self.model_dir)
        return model

    def raw_scores(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return self.model.decision_function(vectors)
