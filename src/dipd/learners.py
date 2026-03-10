import itertools
import logging
import math

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from interpret.glassbox import ExplainableBoostingRegressor
from interpret.utils._clean_x import preclean_X
from interpret.glassbox._ebm._bin import ebm_eval_terms

class Predictor:
    def __init__(self, interactions=0.95, exclude=None):
        self.interactions = interactions
        self.exclude = exclude
        self.model = None

    def predict(self, X, **kwargs):
        fs = sorted(list(X.columns))
        return self.model.predict(X.loc[:, fs], **kwargs)

    def fit(self, X, y):
        fs = sorted(list(X.columns))
        self.model.fit(X.loc[:, fs], y)

    def predict_component(self, X, component):
        pass

    def predict_components(self, X, components):
        return sum([self.predict_component(X, c) for c in components])


class LinearGAM(Predictor):
    def __init__(self, interactions=None, exclude=None):
        if exclude is None:
            exclude = []
        super().__init__(interactions=interactions, exclude=exclude)
        self.model = None
        
    def get_terms(self, X, order=2):
        if self.interactions == 0:
            order = 1
        terms = list([itertools.combinations(X.columns, d) for d in range(1, order+1)])
        terms = list(itertools.chain(*terms))
        terms = [sorted(list(p)) for p in terms]
        if self.exclude is not None:
            terms = [p for p in terms if tuple(p) not in self.exclude]
        return terms
    
    def __check_interactions(self, X, replace_none=True):
        n_interactions = math.comb(X.shape[1], 2)
        n_interactions = n_interactions - len(self.exclude)
        
        if self.interactions is None and replace_none:
            self.interactions = n_interactions
        
        if self.interactions != 0 and self.interactions != n_interactions:
            raise ValueError(
                f'interactions must be 0 or {n_interactions}, got {self.interactions}'
            )
        
    @staticmethod
    def __get_formula(terms):
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
        
    def fit(self, X, y):        
        self.__check_interactions(X, replace_none=True)
        self.terms = self.get_terms(X)
        self.formula = self.__get_formula(self.terms)
        self.model = smf.ols(formula=self.formula,
                             data=pd.concat([X, y], axis=1)).fit()
            
    def predict_component(self, X, component):
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
    
    def __init__(self, interactions=0.95, exclude=None):
        super().__init__(interactions=interactions, exclude=exclude)
        self.model = ExplainableBoostingRegressor(interactions=interactions, exclude=exclude)
                        
    def predict_components(self, X, components):
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
    
    def predict_component(self, X, component):
        return self.predict_components(X, [component])

