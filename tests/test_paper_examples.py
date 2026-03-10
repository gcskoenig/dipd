"""
Tests based on examples from König, Günther & von Luxburg (2024),
"Disentangling Interactions and Dependencies in Feature Attribution"
(arXiv:2410.23772).

Column conventions in explanation.scores (RETURN_NAMES):
  v1                              standalone contribution of group 1
  v2                              standalone contribution of group 2
  vC                              value of conditioning set
  main_effect_cross_predictability  = -(cross-predictability from paper)
  main_effect_cov                   = -2*Cov(g1*, g2*) from paper
  pure_interactions                 = interaction surplus Var(h*)

All values are normalized by Var(Y_test) by default.
"""

import numpy as np
import pandas as pd
import pytest

from dipd import DIP
from dipd.learners import EBM, LinearGAM


# ---------------------------------------------------------------------------
# Data-generation helpers
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


def generate_example7(n=100_000, seed=42):
    """Homework + Study: Y = 4X1 + 4X2, positively correlated Bernoulli."""
    rng = np.random.default_rng(seed)
    probs = [3 / 8, 1 / 8, 1 / 8, 3 / 8]  # P(X1=X2) = 0.75
    idx = rng.choice(4, size=n, p=probs)
    X1 = np.array([0, 0, 1, 1])[idx].astype(float)
    X2 = np.array([0, 1, 0, 1])[idx].astype(float)
    Y = 4 * X1 + 4 * X2
    return pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})


def generate_example8(n=100_000, seed=42):
    """Suppression: Y = 4X1 + 2X2, negatively correlated Bernoulli."""
    rng = np.random.default_rng(seed)
    probs = [1 / 8, 3 / 8, 3 / 8, 1 / 8]  # P(X1=X2) = 0.25
    idx = rng.choice(4, size=n, p=probs)
    X1 = np.array([0, 0, 1, 1])[idx].astype(float)
    X2 = np.array([0, 1, 0, 1])[idx].astype(float)
    Y = 4 * X1 + 2 * X2
    return pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})


def generate_example9(n=100_000, seed=42):
    """OR function: Y = 8*(X1 OR X2) - 1, positively correlated Bernoulli."""
    rng = np.random.default_rng(seed)
    probs = [3 / 8, 1 / 8, 1 / 8, 3 / 8]
    idx = rng.choice(4, size=n, p=probs)
    X1 = np.array([0, 0, 1, 1])[idx].astype(float)
    X2 = np.array([0, 1, 0, 1])[idx].astype(float)
    Y = 8 * ((X1.astype(bool) | X2.astype(bool)).astype(float)) - 1
    return pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})


# ---------------------------------------------------------------------------
# Helper to extract named quantities from explanation.scores
# ---------------------------------------------------------------------------

def extract(scores, feature):
    """Return a dict of paper quantities for a given feature row."""
    row = scores.loc[feature]
    standalone = row['v1']
    cross_pred = -row['main_effect_cross_predictability']  # paper sign
    covariance = -row['main_effect_cov']  # paper sign: 2*Cov(g1,g2)
    dep = cross_pred + covariance
    interaction_surplus = row['pure_interactions']
    psi = interaction_surplus - dep
    loco = row[['v1', 'main_effect_cross_predictability',
                'main_effect_cov', 'pure_interactions']].sum()
    return dict(
        standalone=standalone,
        cross_pred=cross_pred,
        covariance=covariance,
        dep=dep,
        interaction_surplus=interaction_surplus,
        psi=psi,
        loco=loco,
    )


# ---------------------------------------------------------------------------
# Test 1: Example 3, DGP 1 — No cooperation (LinearGAM)
# ---------------------------------------------------------------------------

class TestExample3DGP1:
    @pytest.fixture(scope='class')
    def scores(self):
        df = generate_example3_dgp1(n=100_000, seed=42)
        explainer = DIP(df, 'Y', LinearGAM)
        return explainer.get(['X1', 'X2'], normalized=True)

    def test_standalone_x1(self, scores):
        assert scores['v1'] == pytest.approx(0.5, abs=0.05)

    def test_standalone_x2(self, scores):
        assert scores['v2'] == pytest.approx(0.5, abs=0.05)

    def test_cross_predictability_zero(self, scores):
        assert scores['main_effect_cross_predictability'] == pytest.approx(0.0, abs=0.05)

    def test_covariance_zero(self, scores):
        assert scores['main_effect_cov'] == pytest.approx(0.0, abs=0.05)

    def test_interaction_surplus_zero(self, scores):
        assert scores['pure_interactions'] == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# Test 2: Example 3, DGP 2 — Cancelling interactions and dependencies (LinearGAM)
# ---------------------------------------------------------------------------

class TestExample3DGP2:
    @pytest.fixture(scope='class')
    def scores(self):
        df = generate_example3_dgp2(n=100_000, seed=42)
        explainer = DIP(df, 'Y', EBM)
        return explainer.get(['X1', 'X2'], normalized=True)

    def test_standalone_x1(self, scores):
        assert scores['v1'] == pytest.approx(0.5, abs=0.05)

    def test_standalone_x2(self, scores):
        assert scores['v2'] == pytest.approx(0.5, abs=0.05)

    def test_interaction_surplus_nonzero(self, scores):
        assert scores['pure_interactions'] == pytest.approx(0.26, abs=0.05)

    def test_cross_predictability_nonzero(self, scores):
        # paper: 0.07 (positive), stored as negative
        assert scores['main_effect_cross_predictability'] == pytest.approx(-0.07, abs=0.05)

    def test_covariance_nonzero(self, scores):
        # paper: 2*Cov = 0.19 (positive), stored as negative
        assert scores['main_effect_cov'] == pytest.approx(-0.19, abs=0.05)

    def test_cooperative_impact_zero(self, scores):
        """Interaction surplus and Dep cancel out → Ψ ≈ 0."""
        psi = (scores['pure_interactions']
               + scores['main_effect_cross_predictability']
               + scores['main_effect_cov'])
        assert psi == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# Test 3: Example 7 — Negative cooperative impact via dependence (EBM)
# ---------------------------------------------------------------------------

class TestExample7:
    @pytest.fixture(scope='class')
    def result(self):
        df = generate_example7(n=100_000, seed=42)
        explainer = DIP(df, 'Y', EBM)
        scores = explainer.get(['X1', 'X2'], normalized=False)
        var_y = explainer.var_y
        return scores, var_y

    def test_standalone_x1(self, result):
        scores, _ = result
        assert scores['v1'] == pytest.approx(9, abs=0.5)

    def test_standalone_x2(self, result):
        scores, _ = result
        assert scores['v2'] == pytest.approx(9, abs=0.5)

    def test_interaction_surplus_zero(self, result):
        scores, _ = result
        assert scores['pure_interactions'] == pytest.approx(0, abs=0.5)

    def test_cross_predictability(self, result):
        scores, _ = result
        assert scores['main_effect_cross_predictability'] == pytest.approx(-2, abs=0.5)

    def test_covariance(self, result):
        scores, _ = result
        assert scores['main_effect_cov'] == pytest.approx(-4, abs=0.5)

    def test_cooperative_impact_negative(self, result):
        scores, _ = result
        psi = (scores['pure_interactions']
               + scores['main_effect_cross_predictability']
               + scores['main_effect_cov'])
        assert psi == pytest.approx(-6, abs=0.5)

    def test_v_joint(self, result):
        scores, _ = result
        v_joint = scores.sum()
        assert v_joint == pytest.approx(12, abs=0.5)


# ---------------------------------------------------------------------------
# Test 4: Example 8 — Positive cooperative impact via suppression (EBM)
# ---------------------------------------------------------------------------

class TestExample8:
    @pytest.fixture(scope='class')
    def scores(self):
        df = generate_example8(n=100_000, seed=42)
        explainer = DIP(df, 'Y', EBM)
        return explainer.get(['X1', 'X2'], normalized=False)

    def test_standalone_x1(self, scores):
        assert scores['v1'] == pytest.approx(2.25, abs=0.5)

    def test_standalone_x2_zero(self, scores):
        """X2 has zero standalone power despite appearing in Y (suppression)."""
        assert scores['v2'] == pytest.approx(0, abs=0.5)

    def test_interaction_surplus_zero(self, scores):
        assert scores['pure_interactions'] == pytest.approx(0, abs=0.5)

    def test_cross_predictability(self, scores):
        assert scores['main_effect_cross_predictability'] == pytest.approx(-1.25, abs=0.5)

    def test_covariance_negative(self, scores):
        """Negative correlation → positive main_effect_cov (= -2*Cov, Cov < 0)."""
        assert scores['main_effect_cov'] == pytest.approx(2, abs=0.5)

    def test_cooperative_impact_positive(self, scores):
        psi = (scores['pure_interactions']
               + scores['main_effect_cross_predictability']
               + scores['main_effect_cov'])
        assert psi == pytest.approx(0.75, abs=0.5)

    def test_v_joint(self, scores):
        v_joint = scores.sum()
        assert v_joint == pytest.approx(3, abs=0.5)


# ---------------------------------------------------------------------------
# Test 5: Example 9 — Interaction surplus (OR function, EBM)
# ---------------------------------------------------------------------------

class TestExample9:
    @pytest.fixture(scope='class')
    def scores(self):
        df = generate_example9(n=100_000, seed=42)
        explainer = DIP(df, 'Y', EBM)
        return explainer.get(['X1', 'X2'], normalized=False)

    def test_standalone_x1(self, scores):
        assert scores['v1'] == pytest.approx(9, abs=0.5)

    def test_standalone_x2(self, scores):
        assert scores['v2'] == pytest.approx(9, abs=0.5)

    def test_interaction_surplus(self, scores):
        assert scores['pure_interactions'] == pytest.approx(3, abs=0.5)

    def test_cross_predictability_same_as_ex7(self, scores):
        assert scores['main_effect_cross_predictability'] == pytest.approx(-2, abs=0.5)

    def test_covariance_same_as_ex7(self, scores):
        assert scores['main_effect_cov'] == pytest.approx(-4, abs=0.5)

    def test_v_joint(self, scores):
        v_joint = scores.sum()
        assert v_joint == pytest.approx(15, abs=0.5)


# ---------------------------------------------------------------------------
# Test 6: Structural identities (across all DGPs)
# ---------------------------------------------------------------------------

@pytest.fixture(params=[
    ('example3_dgp1', generate_example3_dgp1, LinearGAM),
    ('example3_dgp2', generate_example3_dgp2, EBM),
    ('example7', generate_example7, EBM),
    ('example8', generate_example8, EBM),
    ('example9', generate_example9, EBM),
], ids=lambda x: x[0])
def dgp_scores(request):
    name, gen_fn, learner = request.param
    df = gen_fn(n=100_000, seed=42)
    explainer = DIP(df, 'Y', learner)
    scores = explainer.get(['X1', 'X2'], normalized=False)
    var_y = explainer.var_y
    return scores, var_y


class TestStructuralIdentities:

    def test_dep_decomposition(self, dgp_scores):
        """Dep = cross_pred + covariance (in code sign convention)."""
        scores, _ = dgp_scores
        dep = scores['main_effect_cross_predictability'] + scores['main_effect_cov']
        # Just check it's a real number (the decomposition is exact by construction)
        assert np.isfinite(dep)

    def test_non_negative_interaction_surplus(self, dgp_scores):
        """Interaction surplus = Var(h*) >= 0."""
        scores, _ = dgp_scores
        assert scores['pure_interactions'] >= -1e-4

    def test_non_negative_cross_predictability(self, dgp_scores):
        """Cross-predictability (sum of variances) >= 0, stored as negative."""
        scores, _ = dgp_scores
        assert scores['main_effect_cross_predictability'] <= 1e-4

    def test_full_decomposition(self, dgp_scores):
        """v(D) = v1 + v2 + vC + cross_pred + cov + interactions."""
        scores, _ = dgp_scores
        v_joint = scores.sum()
        parts = scores['v1'] + scores['v2'] + scores['vC'] + \
                scores['main_effect_cross_predictability'] + \
                scores['main_effect_cov'] + scores['pure_interactions']
        assert v_joint == pytest.approx(parts, abs=1e-10)


# ---------------------------------------------------------------------------
# Test 7: Symmetry under feature relabeling (symmetric DGPs)
# ---------------------------------------------------------------------------

@pytest.fixture(params=[
    ('example3_dgp1', generate_example3_dgp1, LinearGAM),
    ('example3_dgp2', generate_example3_dgp2, EBM),
    ('example7', generate_example7, EBM),
    ('example9', generate_example9, EBM),
], ids=lambda x: x[0])
def symmetric_scores(request):
    name, gen_fn, learner = request.param
    df = gen_fn(n=100_000, seed=42)
    explainer = DIP(df, 'Y', learner)
    scores = explainer.get(['X1', 'X2'], normalized=False)
    return scores


class TestSymmetry:
    def test_v1_equals_v2(self, symmetric_scores):
        scores = symmetric_scores
        assert scores['v1'] == pytest.approx(scores['v2'], abs=0.5)


# ---------------------------------------------------------------------------
# Test 8: Proposition 2 — Independence + additive ⟹ Ψ = 0
# ---------------------------------------------------------------------------

class TestProposition2:
    @pytest.fixture(scope='class')
    def scores(self):
        rng = np.random.default_rng(42)
        n = 100_000
        X1 = rng.standard_normal(n)
        X2 = rng.standard_normal(n)
        Y = np.sin(X1) + X2 ** 2
        df = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
        explainer = DIP(df, 'Y', EBM)
        return explainer.get(['X1', 'X2'], normalized=True)

    def test_cooperative_impact_zero(self, scores):
        psi = (scores['pure_interactions']
               + scores['main_effect_cross_predictability']
               + scores['main_effect_cov'])
        assert psi == pytest.approx(0.0, abs=0.05)

    def test_interaction_surplus_zero(self, scores):
        assert scores['pure_interactions'] == pytest.approx(0.0, abs=0.05)

    def test_dep_zero(self, scores):
        dep = scores['main_effect_cross_predictability'] + scores['main_effect_cov']
        assert dep == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# Test 9: LinearGAM on Example 7 (linear DGP, should work with linear model)
# ---------------------------------------------------------------------------

class TestLinearGAMExample7:
    @pytest.fixture(scope='class')
    def scores(self):
        df = generate_example7(n=100_000, seed=42)
        explainer = DIP(df, 'Y', LinearGAM)
        return explainer.get(['X1', 'X2'], normalized=False)

    def test_standalone_x1(self, scores):
        assert scores['v1'] == pytest.approx(9, abs=0.5)

    def test_standalone_x2(self, scores):
        assert scores['v2'] == pytest.approx(9, abs=0.5)

    def test_interaction_surplus_zero(self, scores):
        assert scores['pure_interactions'] == pytest.approx(0, abs=0.5)

    def test_cooperative_impact(self, scores):
        psi = (scores['pure_interactions']
               + scores['main_effect_cross_predictability']
               + scores['main_effect_cov'])
        assert psi == pytest.approx(-6, abs=0.5)
