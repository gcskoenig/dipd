"""Tests for the variance-first StabilizedEBM learner.

Focus areas:
  - it fits, predicts, and exposes the resolved configuration;
  - the regularization schedule is coarse-by-construction (few bins relative to
    n, heavy bagging) -- the opposite of interpret's high-resolution defaults;
  - ``exclude`` is honored as a hard constraint on the fitted model;
  - the interaction budget is dimensionality/exclude-aware and collapses to a
    purely additive fit when no interactions are allowed;
  - it is a drop-in DIP learner (predict / predict_components work), and a fixed
    random_state makes the decomposition reproducible.
"""
import numpy as np
import pandas as pd

from dipd import DIP, StabilizedEBM


def _make_data(n=120, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    cols = [f'X{i}' for i in range(d)]
    # additive signal plus one genuine interaction between X0 and X1
    y = X[:, 0] + X[:, 1] + 1.5 * X[:, 0] * X[:, 1] + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame(X, columns=cols)
    return df, pd.Series(y, name='Y')


def test_fit_predict_and_params():
    X, y = _make_data()
    model = StabilizedEBM(random_state=0)
    model.fit(X, y)

    assert model.model is not None
    assert model.params_ is not None
    assert 'interactions' in model.params_
    assert 'max_bins' in model.params_

    pred = model.predict(X)
    assert pred.shape == (X.shape[0],)
    r2 = 1 - np.mean((y.values - pred) ** 2) / np.var(y.values)
    assert r2 > 0.5


def test_binning_is_coarse_relative_to_n():
    """The key stability lever: far fewer bins than interpret's default 1024,
    scaled to n so edge bins are not single-point estimates."""
    X, y = _make_data(n=60, d=4)
    model = StabilizedEBM(random_state=0)
    model.fit(X, y)
    # schedule targets ~ n/3 bins, capped by max_bins
    assert model.params_['max_bins'] <= 60 // 3
    assert model.params_['max_bins'] < 1024
    assert model.params_['outer_bags'] >= 16  # heavy bagging by default


def test_smaller_n_gets_fewer_bins():
    """Coarser binning for smaller samples (more regularization when starved)."""
    model = StabilizedEBM()
    few = model._schedule(n=30, d=4)
    many = model._schedule(n=300, d=4)
    assert few['max_bins'] <= many['max_bins']


def test_user_kwargs_override_schedule():
    X, y = _make_data(n=60, d=4)
    # learning_rate is a passthrough estimator kwarg -> exact override;
    # max_bins is an upper cap -> lowers the schedule (n//3 = 20) to 10.
    model = StabilizedEBM(random_state=0, max_bins=10, learning_rate=0.05)
    model.fit(X, y)
    assert model.params_['learning_rate'] == 0.05
    assert model.params_['max_bins'] == 10


def test_exclude_is_respected():
    """An excluded pair must never appear as an interaction term."""
    X, y = _make_data()
    excluded = [('X0', 'X1')]
    model = StabilizedEBM(interactions=10, exclude=excluded, random_state=0)
    model.fit(X, y)

    assert 'X0 & X1' not in model.model.term_names_
    assert model.model.term_names_  # sanity: model actually has terms


def test_budget_collapses_to_additive_when_all_pairs_excluded():
    """If every pair is excluded, the interaction budget must be 0."""
    X, y = _make_data(d=4)
    cols = list(X.columns)
    all_pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]]
    model = StabilizedEBM(exclude=all_pairs, random_state=0)

    assert model._n_allowed_pairs(cols) == 0
    assert model._interaction_budget(0, d=len(cols)) == 0

    model.fit(X, y)
    assert model.params_['interactions'] == 0
    assert all(' & ' not in name for name in model.model.term_names_)


def test_single_feature_has_no_interactions():
    """A univariate fit -- the noisy case StabilizedEBM targets -- has no pairs."""
    X, y = _make_data(d=3)
    model = StabilizedEBM(random_state=0)
    model.fit(X[['X0']], y)
    assert model.params_['interactions'] == 0
    assert all(' & ' not in name for name in model.model.term_names_)


def test_works_as_dip_learner_and_is_reproducible():
    X, y = _make_data(n=150, d=4, seed=1)
    df = pd.concat([X, y], axis=1)
    # a fixed random_state propagates to every sub-model (common random numbers)
    res1 = DIP(df, 'Y', StabilizedEBM, random_state=0).get(['X0', 'X1'])
    res2 = DIP(df, 'Y', StabilizedEBM, random_state=0).get(['X0', 'X1'])
    assert not res1.isnull().any()
    # same seed -> identical decomposition (reproducible)
    pd.testing.assert_series_equal(res1, res2)
    # the X0:X1 interaction in the DGP should surface as positive synergy
    assert res1['pure_interactions'] > 0
