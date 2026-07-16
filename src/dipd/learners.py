import itertools
import logging
import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from interpret.glassbox import ExplainableBoostingRegressor
from interpret.utils._clean_x import preclean_X
from interpret.glassbox._ebm._bin import ebm_eval_terms

class Predictor:
    def __init__(self, interactions: float | None = 0.95,
                 exclude: list[tuple[str, ...]] | None = None,
                 **kwargs: Any) -> None:
        self.interactions = interactions
        self.exclude = exclude
        self.model: Any = None

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        fs = sorted(list(X.columns))
        return self.model.predict(X.loc[:, fs], **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        fs = sorted(list(X.columns))
        self.model.fit(X.loc[:, fs], y)

    def predict_component(self, X: pd.DataFrame,
                          component: str | tuple[str, ...] | list[str]) -> np.ndarray | pd.Series:
        pass

    def predict_components(self, X: pd.DataFrame,
                           components: list[str | tuple[str, ...] | list[str]]) -> np.ndarray | pd.Series:
        return sum([self.predict_component(X, c) for c in components])


class LinearGAM(Predictor):
    def __init__(self, interactions: float | None = None,
                 exclude: list[tuple[str, ...]] | None = None,
                 **kwargs: Any) -> None:
        if exclude is None:
            exclude = []
        super().__init__(interactions=interactions, exclude=exclude)
        self.model = None

    def get_terms(self, X: pd.DataFrame, order: int = 2) -> list[list[str]]:
        if self.interactions == 0:
            order = 1
        terms = list([itertools.combinations(X.columns, d) for d in range(1, order+1)])
        terms = list(itertools.chain(*terms))
        terms = [sorted(list(p)) for p in terms]
        if self.exclude is not None:
            terms = [p for p in terms if tuple(p) not in self.exclude]
        return terms

    def __check_interactions(self, X: pd.DataFrame, replace_none: bool = True) -> None:
        n_interactions = math.comb(X.shape[1], 2)
        n_interactions = n_interactions - len(self.exclude)

        if self.interactions is None and replace_none:
            self.interactions = n_interactions

        if self.interactions != 0 and self.interactions != n_interactions:
            raise ValueError(
                f'interactions must be 0 or {n_interactions}, got {self.interactions}'
            )

    @staticmethod
    def __get_formula(terms: list[list[str] | str]) -> str:
        formula = 'y ~'
        first = True
        for term in terms:
            if isinstance(term, list):
                term = ' * '.join(term)
            if first:
                formula += ' ' + term
                first = False
            else:
                formula += ' + ' + term
        return formula

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        self.__check_interactions(X, replace_none=True)
        self.terms = self.get_terms(X)
        self.formula = self.__get_formula(self.terms)
        self.model = smf.ols(formula=self.formula,
                             data=pd.concat([X, y], axis=1)).fit()

    def predict_component(self, X: pd.DataFrame,
                          component: str | tuple[str, ...] | list[str]) -> pd.Series:
        component_s = component
        if isinstance(component, tuple):
            component_s = list(component_s)
        if isinstance(component_s, list):
            component_s = sorted(component_s)
        if component_s in self.terms:
            if isinstance(component_s, list):
                term = ':'.join(component_s)
            else:
                term = component_s
            coef = self.model.params[term]
            if isinstance(component_s, list):
                prod = X.loc[:, component_s].prod(axis=1)
                return coef * prod
            else:
                return coef * X.loc[:, component_s]
        else:
            return pd.Series(0.0, index=X.index)

class EBM(Predictor):

    def __init__(self, interactions: float | None = 0.95,
                 exclude: list[tuple[str, ...]] | None = None,
                 **kwargs: Any) -> None:
        super().__init__(interactions=interactions, exclude=exclude)
        self.model = ExplainableBoostingRegressor(interactions=interactions, exclude=exclude, **kwargs)

    def predict_components(self, X: pd.DataFrame,
                           components: list[str | tuple[str, ...] | list[str]]) -> np.ndarray:
        """
        Due to limitations of the interpret package we can query multiple components at once,
        but can only get the aggregation of the components at once, not the individual contributions.
        To get the individual contributions we need to query each component separately.
        """
        comp_names = []
        comp_ixs = []
        for component in components:
            if isinstance(component, str):
                comp_name = component
            elif isinstance(component, list) or isinstance(component, tuple):
                component = sorted(component, key=X.columns.tolist().index)
                comp_name = ' & '.join(component)
            else:
                raise NotImplementedError('only str or list of strings supported for component')
            try:
                comp_index = self.model.term_names_.index(comp_name)
                comp_names.append(comp_name)
                comp_ixs.append(comp_index)
            except ValueError as err:
                logging.debug(err)
                logging.debug(f'Probably, component {comp_name} was not found in the model')

        comp_ixs = np.array(comp_ixs).astype(int)

        # taken from the interpret package
        X, n_samples = preclean_X(X, self.model.feature_names_in_, self.model.feature_types_in_)
        n_scores = 1 if isinstance(self.model.intercept_, float) else len(self.model.intercept_)
        explanations = ebm_eval_terms(
            X,
            n_samples,
            n_scores,
            self.model.feature_names_in_,
            self.model.feature_types_in_,
            self.model.bins_,
            [self.model.term_scores_[comp_ix] for comp_ix in comp_ixs],
            [self.model.term_features_[comp_ix] for comp_ix in comp_ixs],
        )

        return np.sum(explanations, axis=1)

    def predict_component(self, X: pd.DataFrame,
                          component: str | tuple[str, ...] | list[str]) -> np.ndarray:
        return self.predict_components(X, [component])


class StabilizedEBM(EBM):
    """A variance-first EBM for stable DIP decompositions.

    Motivation
    ----------
    In a LOO DIP decomposition the reported scores are *differences* of value
    functions ``v(S)`` (explained test MSE), and every EBM fit is stochastic
    (bootstrap bagging + the internal early-stopping validation split). On small
    data this refit variance is large and -- empirically on the Cambodia
    nowcasting data -- it is dominated by the *univariate* (single-feature)
    sub-models: interpret's default ``max_bins=1024`` puts ~1 training row in each
    extreme bin, so a lone shape function's tails are essentially single-point
    estimates and swing wildly from seed to seed, especially for trending features
    whose test points sit at the edge of the training range.

    Unlike a tuned EBM, this learner does **not** search hyperparameters -- a
    search on ~50 rows is itself high-variance and can pick *less* regularization
    for exactly those fragile single-feature fits. Instead it applies a fixed,
    sample-size-aware *regularization* schedule with three bias-safe (or
    bias-cheap) levers:

      1. Coarse binning -- ``max_bins`` is capped relative to ``n`` so edge bins
         hold several rows and their scores stop being single-point estimates.
         This is the dominant fix for the univariate-fit noise.
      2. Heavy bagging -- ``outer_bags`` is set high to average out the bootstrap
         and validation-split randomness (refit std shrinks ~ ``1/sqrt(bags)``).
      3. Strong shape smoothing -- large ``smoothing_rounds``, shallow trees
         (``max_leaves=2``), a small ``learning_rate`` and a ``min_samples_leaf``
         floor scaled to ``n``.

    A fourth lever -- *common random numbers* across the differenced sub-models --
    is handled by ``DIP`` itself: when the explainer is given a ``random_state`` it
    propagates the *same* seed to every sub-model, so their shared bagging noise
    cancels in ``v_full - v_GAM`` and ``v_GAM - v_f1 - v_f2``. This learner honors
    that seed by passing it through to the underlying estimator.

    As in the DIP protocol, ``exclude`` is a hard constraint (never tuned) and the
    interaction budget is capped by the number of *allowed* (non-excluded) pairs,
    so a restricted sub-model with no allowed pairs degrades to a purely additive
    fit. The budget is also kept modest (``<= d``) on small ``n`` to avoid the
    many-interaction overfitting of interpret's ``'5x'`` / 0.95 default.

    Any explicitly passed estimator keyword (e.g. ``learning_rate=0.05``)
    overrides the schedule. ``max_bins`` and ``outer_bags`` are exposed as
    dedicated arguments: ``outer_bags`` sets the bag count directly, while
    ``max_bins`` acts as an upper *cap* that the sample-size schedule may lower.

    Parameters
    ----------
    interactions:
        Upper bound on the interaction budget, in interpret's float/int
        convention (``None`` = all allowed pairs). Always capped by the number of
        non-excluded pairs and by ``d``.
    exclude:
        Interaction terms forbidden for this fit (hard constraint).
    outer_bags:
        Number of outer bags for variance reduction (default 32).
    max_bins:
        Upper cap on ``max_bins``; the schedule may lower it based on ``n``.
    random_state:
        Seed passed through to the estimator (and, via ``DIP``, shared across
        sub-models for common-random-number variance reduction).

    Attributes set after ``fit``:
        params_: the resolved hyperparameter dict actually passed to the EBM.
    """

    def __init__(self, interactions: float | int | None = None,
                 exclude: list[tuple[str, ...]] | None = None,
                 outer_bags: int = 32, max_bins: int = 256,
                 random_state: int | None = None, **kwargs: Any) -> None:
        # Bypass EBM.__init__ (which eagerly builds a default-hyperparameter
        # estimator); the model is built in fit() once n and d are known.
        Predictor.__init__(self, interactions=interactions, exclude=exclude)
        self.outer_bags = outer_bags
        self.max_bins = max_bins          # upper cap; the schedule may lower it
        self.random_state = random_state
        self.fixed_kwargs = dict(kwargs)
        if random_state is not None and 'random_state' not in self.fixed_kwargs:
            self.fixed_kwargs['random_state'] = random_state
        self.model = None
        self.params_: dict[str, Any] | None = None

    def _n_allowed_pairs(self, columns: list[str]) -> int:
        """Number of pairwise interactions still permitted, i.e. all pairs minus
        those forbidden by ``exclude``."""
        d = len(columns)
        n_pairs = math.comb(d, 2)
        cols = set(columns)
        excl = self.exclude or []
        n_excluded = sum(1 for t in excl if len(t) == 2 and all(f in cols for f in t))
        return max(0, n_pairs - n_excluded)

    def _user_interaction_cap(self, n_allowed: int) -> int:
        """Resolve a user-supplied ``interactions`` value into an upper bound on
        the interaction budget, following interpret's float/int convention."""
        iv = self.interactions
        if iv is None:
            return n_allowed
        if isinstance(iv, float) and 0.0 < iv <= 1.0:
            return min(n_allowed, max(0, round(iv * n_allowed)))
        return min(n_allowed, int(iv))

    def _schedule(self, n: int, d: int) -> dict[str, Any]:
        """Fixed, sample-size-aware regularization. The binning cap is the key
        knob: we want several rows per bin so the shape function's tails are not
        single-point estimates. ``max_bins`` therefore scales with ``n`` (about
        ``n / 3`` bins) rather than the interpret default of 1024."""
        max_bins = int(max(8, min(self.max_bins, n // 3)))
        min_samples_leaf = int(max(4, round(n / 12)))
        return dict(
            max_bins=max_bins,
            learning_rate=0.01,
            max_leaves=2,
            min_samples_leaf=min_samples_leaf,
            smoothing_rounds=500,
            outer_bags=self.outer_bags,
        )

    def _interaction_budget(self, n_allowed: int, d: int) -> int:
        """Interaction count: capped by the allowed pairs, by any user request,
        and kept modest (``<= d``) on small data to avoid overfitting. Collapses
        to 0 for a single feature or a fully cross-excluded sub-model."""
        cap = min(n_allowed, self._user_interaction_cap(n_allowed))
        if cap == 0:
            return 0
        return int(min(cap, max(1, d)))

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        fs = sorted(list(X.columns))
        X = X.loc[:, fs]
        y = np.asarray(y)
        n, d = X.shape

        params = self._schedule(n, d)
        params['interactions'] = self._interaction_budget(self._n_allowed_pairs(fs), d)
        # user-supplied kwargs win over the schedule; exclude is passed separately
        merged = {**params, **self.fixed_kwargs}
        self.params_ = merged
        self.model = ExplainableBoostingRegressor(exclude=self.exclude, **merged)
        self.model.fit(X, y)
