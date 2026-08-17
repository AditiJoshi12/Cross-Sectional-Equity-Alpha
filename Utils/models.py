"""
Model factories for the walk-forward harness.

Each factory returns a fresh estimator. The target is in raw return units (~1e-4),
so the linear models are wrapped in TransformedTargetRegressor to rescale to bps
before fitting — otherwise L1/L2 penalties shrink everything to zero.

Progression: Ridge (linear baseline) -> ElasticNet (sparse linear) -> GBRT
(nonlinear). Penalty strength is chosen by inner CV, not hardcoded.
"""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor

_BPS = 1e4


def _to_bps(y):
    return y * _BPS


def _from_bps(y):
    return y / _BPS


def ridge_factory():
    base = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 4, 13)))
    return TransformedTargetRegressor(base, func=_to_bps, inverse_func=_from_bps)


def elasticnet_factory():
    base = make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-3, 1, 12),
                     max_iter=20000, cv=3))
    return TransformedTargetRegressor(base, func=_to_bps, inverse_func=_from_bps)


def gbrt_factory():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            max_depth=3, min_child_weight=200,
            n_estimators=300, learning_rate=0.02,
            subsample=0.6, colsample_bytree=0.5,
            reg_alpha=0.1, reg_lambda=1.0,
            objective="reg:squarederror", n_jobs=-1)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=300, max_depth=3,
            learning_rate=0.02, subsample=0.6,
            min_samples_leaf=200)


def make_model_factories():
    return {"ridge": ridge_factory,
            "elasticnet": elasticnet_factory,
            "gbrt": gbrt_factory}
