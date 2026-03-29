"""Tests for the SplineGAM learner."""

import numpy as np
import pandas as pd
import pytest

from dipd import DIP
from dipd.learners import SplineGAM


# ---------------------------------------------------------------------------
# Data-generation helpers (same as test_paper_examples.py)
# ---------------------------------------------------------------------------

def generate_example3_dgp1(n=100_000, seed=42):
    """Y = X1 + X2, X ~ N(0, I). No interactions, no dependencies."""
    rng = np.random.default_rng(seed)
    X1 = rng.standard_normal(n)
    X2 = rng.standard_normal(n)
    Y = X1 + X2
    return pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})


def generate_example3_dgp2(n=100_000, seed=42):
    """Y = X1 + X2 + sqrt(6)*X1*X2, X ~ N(0, [[1,0.5],[0.5,1]])."""
    rng = np.random.default_rng(seed)
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    X = rng.multivariate_normal(mean, cov, size=n)
    X1, X2 = X[:, 0], X[:, 1]
    Y = X1 + X2 + np.sqrt(6) * X1 * X2
    return pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestSplineGAMUnit:
    """Unit tests for SplineGAM construction and term building."""

    def test_main_effects_only(self):
        """interactions=0 should produce only s() terms, no te() terms."""
        gam = SplineGAM(interactions=0)
        df = generate_example3_dgp1(n=1000)
        X, y = df[['X1', 'X2']], df['Y']
        gam.fit(X, y)

        # term_map should have only main-effect keys (strings)
        for key in gam._term_map:
            assert isinstance(key, str), f"Expected string key, got {type(key)}: {key}"
        assert 'X1' in gam._term_map
        assert 'X2' in gam._term_map

    def test_all_interactions(self):
        """interactions=None should include all pairwise te() terms."""
        gam = SplineGAM(interactions=None)
        df = generate_example3_dgp1(n=1000)
        X, y = df[['X1', 'X2']], df['Y']
        gam.fit(X, y)

        assert 'X1' in gam._term_map
        assert 'X2' in gam._term_map
        assert ('X1', 'X2') in gam._term_map

    def test_exclude(self):
        """Excluded pairs should not appear in the term map."""
        gam = SplineGAM(interactions=None, exclude=[('X1', 'X2')])
        df = generate_example3_dgp1(n=1000)
        X, y = df[['X1', 'X2']], df['Y']
        gam.fit(X, y)

        assert ('X1', 'X2') not in gam._term_map
        assert 'X1' in gam._term_map
        assert 'X2' in gam._term_map

    def test_fractional_interactions(self):
        """interactions=0.5 with 3 features (3 possible pairs) keeps ~1-2 pairs."""
        rng = np.random.default_rng(42)
        n = 1000
        X = pd.DataFrame({
            'A': rng.standard_normal(n),
            'B': rng.standard_normal(n),
            'C': rng.standard_normal(n),
        })
        y = X['A'] + X['B'] + X['C']

        gam = SplineGAM(interactions=0.5)
        gam.fit(X, y)

        interaction_keys = [k for k in gam._term_map if isinstance(k, tuple)]
        # 3 pairs * 0.5 = 1.5, rounded to 2
        assert len(interaction_keys) == 2

    def test_decomposition_property(self):
        """Sum of all component contributions + intercept should equal predict."""
        gam = SplineGAM(interactions=None, n_splines=10)
        df = generate_example3_dgp1(n=2000, seed=99)
        X, y = df[['X1', 'X2']], df['Y']
        gam.fit(X, y)

        X_test = X.iloc[:100]
        preds = gam.predict(X_test)

        # Sum all components
        total = np.zeros(len(X_test))
        for key in gam._term_map:
            component = key if isinstance(key, str) else list(key)
            total += gam.predict_component(X_test, component)

        # Add intercept (scalar coefficient for the intercept term)
        total += gam.model.coef_[-1]  # pyGAM stores intercept as last coef

        np.testing.assert_allclose(total, preds, atol=1e-6,
                                   err_msg="Component sum + intercept != predict")

    def test_missing_component_returns_zeros(self):
        """Querying a component not in the model should return zeros."""
        gam = SplineGAM(interactions=0)
        df = generate_example3_dgp1(n=1000)
        X, y = df[['X1', 'X2']], df['Y']
        gam.fit(X, y)

        result = gam.predict_component(X.iloc[:10], ('X1', 'X2'))
        np.testing.assert_array_equal(result, 0.0)

    def test_predict_components_sum(self):
        """predict_components should sum individual predict_component calls."""
        gam = SplineGAM(interactions=None, n_splines=10)
        df = generate_example3_dgp1(n=1000)
        X, y = df[['X1', 'X2']], df['Y']
        gam.fit(X, y)

        X_test = X.iloc[:50]
        combined = gam.predict_components(X_test, ['X1', ('X1', 'X2')])
        individual = (gam.predict_component(X_test, 'X1')
                      + gam.predict_component(X_test, ('X1', 'X2')))
        np.testing.assert_allclose(combined, individual, atol=1e-10)


# ---------------------------------------------------------------------------
# Integration tests with DIP
# ---------------------------------------------------------------------------

class TestSplineGAMIntegration:
    """Integration tests: SplineGAM inside the DIP decomposition."""

    def test_example3_dgp1(self):
        """Y = X1 + X2, independent features.

        Expected: standalone ~ 0.5 each, all cooperation terms ~ 0.
        """
        df = generate_example3_dgp1(n=100_000, seed=42)

        explainer = DIP(df, 'Y', SplineGAM,
                        learner_kwargs={'n_splines': 25})
        scores = explainer.get(['X1', 'X2'], normalized=True)

        # Standalone contributions should each be ~0.5
        assert scores['v1'] == pytest.approx(0.5, abs=0.05)
        assert scores['v2'] == pytest.approx(0.5, abs=0.05)

        # Cooperation terms should be ~0
        assert scores['main_effect_cross_predictability'] == pytest.approx(0.0, abs=0.05)
        assert scores['main_effect_cov'] == pytest.approx(0.0, abs=0.05)
        assert scores['pure_interactions'] == pytest.approx(0.0, abs=0.05)

    def test_example3_dgp2(self):
        """Y = X1 + X2 + sqrt(6)*X1*X2, correlated features.

        Expected: interaction surplus (pure_interactions) ~ 0.26.
        """
        df = generate_example3_dgp2(n=100_000, seed=42)

        explainer = DIP(df, 'Y', SplineGAM,
                        learner_kwargs={'n_splines': 25})
        scores = explainer.get(['X1', 'X2'], normalized=True)

        assert scores['pure_interactions'] == pytest.approx(0.26, abs=0.08)
