from typing import List, Optional
import numpy as np
import joblib
import catboost
from catboost import CatBoostRanker, Pool
import lightgbm as lgb

def predict_ensemble(
    lgbm_models: Optional[List] = None,
    cat_models: Optional[List] = None,
    X=None,
    cat_features: Optional[List[str]] = None,
    lgbm_weight: float = 0.6,
) -> np.ndarray:

    n = len(X)
    lgbm_score = np.zeros(n, dtype=np.float32)
    cat_score = np.zeros(n, dtype=np.float32)

    if lgbm_models:
        for m in lgbm_models:
            lgbm_score += m.predict(X)
        lgbm_score /= len(lgbm_models)

    if cat_models:
        pool = Pool(data=X, cat_features=cat_features or [])
        for m in cat_models:
            cat_score += m.predict(pool)
        cat_score /= len(cat_models)

    if lgbm_models and cat_models:
        final = (lgbm_weight * lgbm_score) + ((1.0 - lgbm_weight) * cat_score)
    elif lgbm_models:
        final = lgbm_score
    elif cat_models:
        final = cat_score
    return final


def save_model(obj, path: str) -> None:
    if hasattr(obj, 'save_model') and not path.endswith('.pkl'):
        obj.save_model(path)
    else:
        joblib.dump(obj, path)


def load_lgbm_models(paths: List[str]):
    return [joblib.load(p) for p in paths]


def load_cat_models(paths: List[str]):
    models = []
    for p in paths:
        m = CatBoostRanker()
        m.load_model(p)
        models.append(m)
    return models
