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

logger = logging.getLogger(__name__)

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


class SplineGAM(Predictor):
    """Spline-based GAM learner using pyGAM.

    Fits a generalized additive model with spline main effects and optional
    tensor-product interaction terms via the ``pygam`` library.
    """

    def __init__(self, interactions: float | None = 0.95,
                 exclude: list[tuple[str, ...]] | None = None,
                 n_splines: int = 25,
                 **kwargs: Any) -> None:
        super().__init__(interactions=interactions, exclude=exclude)
        self.n_splines = n_splines
        self.kwargs = kwargs
        self._name_to_idx: dict[str, int] = {}
        self._term_map: dict[str | tuple[str, ...], int] = {}

    @staticmethod
    def _import_pygam():  # noqa: ANN205
        try:
            import pygam
            return pygam
        except ImportError:
            raise ImportError(
                "SplineGAM requires pygam. Install with: pip install dipd[spline]"
            ) from None

    def _build_terms(self, feature_names: list[str]):  # noqa: ANN205
        """Build pygam TermList from feature names and interaction settings."""
        pygam = self._import_pygam()

        terms = None
        for name in feature_names:
            idx = self._name_to_idx[name]
            t = pygam.s(idx, n_splines=self.n_splines)
            terms = t if terms is None else terms + t

        if self.interactions != 0:
            pairs = list(itertools.combinations(feature_names, 2))

            if self.exclude:
                exclude_set = {tuple(sorted(e)) for e in self.exclude}
                pairs = [p for p in pairs if tuple(sorted(p)) not in exclude_set]

            if isinstance(self.interactions, float) and 0 < self.interactions <= 1:
                n_keep = max(1, int(round(self.interactions * len(pairs))))
                pairs = pairs[:n_keep]
            # interactions=None means keep all remaining pairs

            for a, b in pairs:
                t = pygam.te(self._name_to_idx[a], self._name_to_idx[b],
                             n_splines=self.n_splines)
                terms = terms + t

        return terms

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        pygam = self._import_pygam()

        fs = sorted(list(X.columns))
        self._name_to_idx = {name: i for i, name in enumerate(fs)}

        terms = self._build_terms(fs)
        self.model = pygam.LinearGAM(terms, **self.kwargs)
        self.model.fit(X.loc[:, fs].values, np.asarray(y))

        # Build term map: pygam term index -> component key
        self._term_map = {}
        idx_to_name = {v: k for k, v in self._name_to_idx.items()}
        for term_idx, term in enumerate(self.model.terms):
            term_type = term.info.get('term_type', '')
            if term_type == 'intercept_term':
                continue
            feature = term.feature
            if feature is None:
                continue
            if isinstance(feature, (list, tuple)):
                names = [idx_to_name[int(fi)] for fi in feature]
                self._term_map[tuple(sorted(names))] = term_idx
            else:
                self._term_map[idx_to_name[int(feature)]] = term_idx

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        fs = sorted(list(X.columns))
        return self.model.predict(X.loc[:, fs].values)

    def predict_component(self, X: pd.DataFrame,
                          component: str | tuple[str, ...] | list[str]) -> np.ndarray:
        fs = sorted(list(X.columns))
        X_array = X.loc[:, fs].values

        # Normalize component to lookup key
        if isinstance(component, list):
            key: str | tuple[str, ...] = tuple(sorted(component))
        elif isinstance(component, tuple):
            key = tuple(sorted(component))
        else:
            key = component

        term_idx = self._term_map.get(key)
        if term_idx is None:
            logger.debug(f"Component {key} not found in model terms, returning zeros")
            return np.zeros(X.shape[0])

        return self.model.partial_dependence(term=term_idx, X=X_array)
