import numpy as np
import pandas as pd
import scipy.special 
import math
import tqdm
import itertools
import logging
import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from kollabi.plots import forceplot

interpret_logger = logging.getLogger('interpret')
interpret_logger.setLevel(logging.WARNING)

idx = pd.IndexSlice


class CollabExplainer:
    """
    A class for computing feature decompositions and collaboration measures in a dataset.

    Parameters:
        df (pandas.DataFrame): The input dataset.
        target (str): The target variable name.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """

    RETURN_NAMES = ['var_g1', 'var_g2', 'var_gC', 'additive_collab_explv', 'additive_collab_cov', 'interactive_collab']

    def __init__(self, df, target, learner, test_size=0.2, verbose=False) -> None:
        self.df = df
        self.target = target
        self.fs = [col for col in df.columns if col != target]
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df[self.fs], df[target], test_size=test_size)
        self.verbose = verbose
        self.decomps = {}
        self.Learner = learner
        self.models = {}
        
    def new_split(self, test_size=None):
            """
            Splits the dataset into training and testing sets.

            Args:
                test_size (float, optional): The proportion of the dataset to include in the test split.
                    If not specified, the default test size defined in the class will be used.

            Returns:
                None

            Raises:
                None
            """
            if test_size is None:
                test_size = self.test_size
            else:
                self.test_size = test_size
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df[self.fs], self.df[self.target],
                                                                                    test_size=test_size)
            self.decomps.clear()
            
    # @staticmethod
    # def _sort_fs(fs):
    #     return tuple(sorted(fs))
        
    @staticmethod
    def __sort_comb(comb, inner_only=False):
        """
        Sorts the combinations in the given `comb` tuple.

        Args:
            comb (list): The list of combinations to be sorted.
            inner_only (bool, optional): If True, only the inner elements will be sorted. 
                If False, first the inner elements, then the outer tuple will be sorted..

        Returns:
            tuple: The sorted combinations.
        """
        # TODO accomodate for a conditioning group, make sure that the conditioning is always last but the rest is sorted
        comb_s = tuple(tuple(sorted(c)) for c in comb)
        if not inner_only:
            comb_s = tuple(sorted(comb_s, key=lambda x: (len(x), x)))
            comb_s = tuple(tuple(gr) for gr in comb_s)
        return comb_s
    
    @staticmethod
    def __make_tuple(comb):
        return tuple([tuple(gr) for gr in comb])

    @staticmethod
    def __adjust_order(comb, res):
        if CollabExplainer.__sort_comb(comb, inner_only=True) == CollabExplainer.__sort_comb(comb, inner_only=False):
            return res
        else:
            res_s = res.rename({CollabExplainer.RETURN_NAMES[0]: CollabExplainer.RETURN_NAMES[1],
                                CollabExplainer.RETURN_NAMES[1]: CollabExplainer.RETURN_NAMES[0]}, inplace=False).copy()
            res_s = res_s.loc[res.index]
            return res_s
        
    @staticmethod
    def __get_terms(fs, order, exclude=[]):
        terms = sum([list(itertools.combinations(fs, d)) for d in range(2, order+1)], [])
        terms = [term for term in terms if term not in exclude]
        # if order >= 1:
        #     terms += fs
        return terms
    
    @staticmethod
    def __get_excluded_terms(comb, order, C=[]):
        """
        Get the terms that are not in the combination.
        """
        C_l = list(C)
        termss = [CollabExplainer.__get_terms(elem + C_l, order) for elem in comb]
        allowed_terms = list(itertools.chain(*termss))
        fs = [f for gr in comb for f in gr] + C_l
        all_terms = CollabExplainer.__get_terms(fs, order)
        return [term for term in all_terms if term not in allowed_terms]
        
    def __get_model(self, comb, order, C=[]):
        """
        The comb tuple of tuples indicates which groups of features are allowed to interact.
        The order specifies the max order of interactions.
        So if we get a tuple with one tuple containing all features, we get the full model of order òrder`.
        If we get a tuple with two tuples, we get the model of order `order` with the 
        interactions within the groups only.
        The `C` parameter can be used to specify a set of features that we assume to be "known" before,
        i.e. we fit the model on the residual of the best model with the features C.
        Those features and all interactions involving the features can also be used by the model.
        """
        # add conditioning set
        comb_s = CollabExplainer.__sort_comb(comb)
        C_s = sorted(list(C))
        fs = [f for gr in comb for f in gr]
        fs_full = fs + C
                
        key = (order, comb_s, tuple(C_s))
        if key in self.models.keys():
            logging.debug(f'Using precomputed model for {comb_s}')
            return self.models[key]
        else:
            logging.debug(f'Fitting model for {comb_s}')
            
            # regress out conditioning set if nonempty
            if len(C_s) > 0:
                model = self.__get_model([C_s], order, C=[])
                y_pred = model.predict(self.X_train.loc[:, C_s])
                y_res = self.y_train - y_pred
            else:
                y_res = self.y_train
                
            # specify model and the allowed interactions
            if len(comb_s) > 1:
                excluded_terms = CollabExplainer.__get_excluded_terms(comb, order, C=C)
                model = self.Learner(exclude=excluded_terms)
            else:
                model = self.Learner(exclude=None)
                
            model.fit(self.X_train.loc[:, fs_full], y_res)
            self.models[key] = model
            return model
                
    def __assert_comb_valid(self, comb, C=[]):
        """
        Asserts that the combination contains two elements, that the features are in the columns, that the 
        two sets are disjoint. If an element is a string, it is converted to a list, such that always a list
        of two lists is returned.
        """
        assert len(comb) == 2, 'Please provide exactly two sets of features'
        comb_ = list(comb)
        for i in range(len(comb_)):
            if isinstance(comb_[i], str):
                comb_[i] = [comb_[i]]
            elif isinstance(comb_[i], tuple):
                comb_[i] = list(comb_[i])
            else:
                assert isinstance(comb_[i], list), 'The elements of the combination must be strings or lists'
            assert all([f in self.fs for f in comb_[i]]), 'Feature not in the dataset'
        assert len(set(comb_[0]).intersection(set(comb_[1]))) == 0, 'the two sets of features must be disjoint'
        assert len(set(comb_[0]).union(set(comb_[1])).intersection(set(C))) == 0, 'the conditioning set must be disjoint from the two sets'
        return comb_
                    
    def get(self, comb, C=[]):
        comb = self.__assert_comb_valid(comb)
        comb_s = CollabExplainer.__sort_comb(comb)
        key = (comb_s, tuple(sorted(C)))
        if key in self.decomps.keys():
            res = self.decomps[key]
            return self.__adjust_order(comb, res)
        else:
            res = self.compute(list(comb_s), C=C)
            self.decomps[key] = res
            return res
        
    def compute(self, comb, order=2, C=[]):
        """
        Computes decomposition for a combination comb conditional on
        a group of features C. Uses GAMs of at most order `order`
        to compute the decomposition.
        
        Args:
            comb (list): A list of two lists of features.
            order (int, optional): The maximum order of interactions. Defaults to 2.
            C (list, optional): A list of features that are assumed to be known. Defaults to [].
        """
        comb = self.__assert_comb_valid(comb)
        return_names = self.RETURN_NAMES
        
        # comb = [f for gr in comb for f in gr]
        fs = [f for gr in comb for f in gr]
        fs_full = fs + C
        fs_0 = comb[0] + C
        fs_1 = comb[1] + C
        
        # get baseline
        if len(C) > 0:
            b = self.__get_model([C], order, C=[])
            b_pred = b.predict(self.X_test.loc[:, C])
        else:
            b_pred = 0
        y_test_res = self.y_test - b_pred 
        var_y_res = np.var(y_test_res)
        var_fC = np.var(b_pred) / var_y_res
        
        start = time.time()
        model_full = self.__get_model([fs], order, C=C)
        end = time.time()
        logging.info(f'Fitting full model took {end-start} seconds')
        # model_full = self.Learner(interactions=scipy.special.comb(len(all_fs), order))
        # model_full.fit(self.X_train.loc[:, all_fs], self.y_train)
        var_total = r2_score(y_test_res, model_full.predict(self.X_test[fs_full]))

        start = time.time()
        model_order1 = self.__get_model(comb, order, C=C)
        end = time.time()
        logging.info(f'Fitting model without interactions between the groups took {end-start} seconds')        
        # model_order1 = self.Learner(interactions=terms_g)
        # model_order1.fit(self.X_train.loc[:, all_fs], self.y_train)
        var_GAM = r2_score(y_test_res, model_order1.predict(self.X_test[fs_full]))

        start = time.time()
        f1 = self.__get_model([comb[0]], order, C=C)
        end = time.time()
        logging.info(f'Fitting model for group 1 took {end-start} seconds')
        # f1 = self.Learner(interactions=scipy.special.comb(len(comb[0]), order))
        # f1.fit(self.X_train[comb[0]], self.y_train)
        var_f1 = r2_score(y_test_res, f1.predict(self.X_test[fs_0]))

        start = time.time()
        f2 = self.__get_model([comb[1]], order, C=C)
        end = time.time()
        logging.info(f'Fitting model for group 2 took {end-start} seconds')
        # f2 = self.Learner(interactions=scipy.special.comb(len(comb[1]), order))
        # f2.fit(self.X_train[comb[1]], self.y_train)
        var_f2 = r2_score(y_test_res, f2.predict(self.X_test[fs_1]))

        # get the GAM components
        terms_C = self.__get_terms(C, order) + C
        terms_g1 = self.__get_terms(fs_0, order, exclude=terms_C) + comb[0]
        terms_g2 = self.__get_terms(fs_1, order, exclude=terms_C) + comb[1]
                
        # get the GAM components
        g1_test = model_order1.predict_components(self.X_test, terms_g1)
        g2_test = model_order1.predict_components(self.X_test, terms_g2)        
        
        # if C is not empty, we make the GAM components orthogonal to C to recover uniquness
        if len(C) > 0:
            g1_train = model_order1.predict_components(self.X_train, terms_g1)
            g2_train = model_order1.predict_components(self.X_train, terms_g2)
            
            model_g1 = self.Learner()
            model_g1.fit(self.X_train[C], g1_train)
            g1_pred = model_g1.predict(self.X_test[C])
            g1_res = g1_test - g1_pred
            model_g2 = self.Learner()
            model_g2.fit(self.X_train[C], g2_train)
            g2_pred = model_g2.predict(self.X_test[C])
            g2_res = g2_test - g2_pred
            
            gc_test = model_order1.predict_components(self.X_test, terms_C)
            var_comp = np.var(g1_pred + g2_pred + gc_test) / np.var(g1_test + g2_test + gc_test)
            logging.debug(f'Variance of GAM explained by X_C: {var_comp}')
        else:
            g1_res = g1_test
            g2_res = g2_test
                
        cov_g1_g2 = np.cov(g1_res, g2_res)[0, 1]
        cov_g1_g2 = cov_g1_g2 / var_y_res
        additive_collab = (var_f1 + var_f2 - var_GAM)*-1
        additive_collab_wo_cov = additive_collab + 2*cov_g1_g2            
        interactive_collab = var_total - var_GAM

        if self.verbose:
            print(comb)
            print('total variance Y ', var_y_res)
            print('test v(1 cup 2)', var_total)
            print('training v(1 cup 2): ', r2_score(self.y_train, model_full.predict(self.X_train)))
            print('Interactive Collaboration: ', interactive_collab)
            print('v(', comb[0], '): ', var_f1)
            print('v(', comb[1], '): ', var_f2)

            print('Cov(g1, g2): ', cov_g1_g2 / var_y_res)
            print('Additive Collaboration: ', additive_collab)
         
        # rescale to proportion of variance of Y 
        if True:
            var_y = np.var(self.y_test)
            factor = var_y_res / var_y
            var_f1 *= factor
            var_f2 *= factor
            additive_collab *= factor
            cov_g1_g2 *= factor
            interactive_collab *= factor
            var_fC *= factor
                   
        return pd.Series([var_f1, var_f2, var_fC, additive_collab_wo_cov, -2*cov_g1_g2, interactive_collab], index=return_names) 
        
    def get_all_pairwise(self, only_precomputed=False, return_matrixs=False):
        '''
        Gives a detailed decomposition of all features respecting interactions and the dependencies between them

        Parameters:
            data: A datasetcontaining all the data
        '''
        logging.info('Computing all decompositions')
        if only_precomputed:
            combinations = list(self.decomps.keys())
        else:
            combinations = [list(comb) for comb in itertools.combinations(self.fs, 2)]
        
        if return_matrixs:
            vars_bivarivate = pd.DataFrame(index=self.fs, columns=self.fs, dtype=float)
            additive_collab = pd.DataFrame(index=self.fs, columns=self.fs, dtype=float)
            neg2_cov_g1_g2 = pd.DataFrame(index=self.fs, columns=self.fs, dtype=float)
            additive_collab_wo_cov = pd.DataFrame(index=self.fs, columns=self.fs, dtype=float)
            synergetic_collab = pd.DataFrame(index=self.fs, columns=self.fs, dtype=float)
            
            for comb in tqdm.tqdm(combinations):
                res = self.get(comb)
                # hacky but works
                vars_bivarivate.loc[comb[0], comb[0]] = res[self.RETURN_NAMES[0]]
                vars_bivarivate.loc[comb[1], comb[1]] = res[self.RETURN_NAMES[1]]  
                # rest                              
                vars_bivarivate.loc[comb[0], comb[1]] = res.sum(axis=0)
                additive_collab.loc[comb[0], comb[1]] = res[self.RETURN_NAMES[3]] + res[self.RETURN_NAMES[4]]
                neg2_cov_g1_g2.loc[comb[0], comb[1]] = res[self.RETURN_NAMES[4]]
                additive_collab_wo_cov.loc[comb[0], comb[1]] = res[self.RETURN_NAMES[3]]
                synergetic_collab.loc[comb[0], comb[1]] = res[self.RETURN_NAMES[5]]
                # make symmetric
                vars_bivarivate.loc[comb[1], comb[0]] = vars_bivarivate.loc[comb[0], comb[1]]
                additive_collab.loc[comb[1], comb[0]] = additive_collab.loc[comb[0], comb[1]]
                neg2_cov_g1_g2.loc[comb[1], comb[0]] = neg2_cov_g1_g2.loc[comb[0], comb[1]]
                additive_collab_wo_cov.loc[comb[1], comb[0]] = additive_collab_wo_cov.loc[comb[0], comb[1]]
                synergetic_collab.loc[comb[1], comb[0]] = synergetic_collab.loc[comb[0], comb[1]]
            
            return vars_bivarivate, additive_collab, synergetic_collab, neg2_cov_g1_g2, additive_collab_wo_cov
        else:                
            results = pd.DataFrame(combinations, columns=['feature1', 'feature2'])
            results.set_index(['feature1', 'feature2'], inplace=True)
            for comb in tqdm.tqdm(combinations):
                res = self.get(comb)
                results.loc[tuple(comb), res.index] = res
                res2 = self.get(comb[::-1])
                results.loc[tuple(comb[::-1]), res2.index] = res2
            return results
        
    def get_all_pairwise_onefixed(self, feature):
        '''
        Gives a detailed decomposition of all features respecting interactions and the dependencies between them

        Parameters:
            data: A datasetcontaining all the data
        '''
        logging.info(f'Computing all decompositions for feature {feature}')
        combinations = [[feature, col] for col in self.fs if col != feature]
        results = pd.DataFrame(combinations, columns=['feature1', 'feature2'])
        results.set_index(['feature1', 'feature2'], inplace=True)
        for comb in tqdm.tqdm(combinations):
            res = self.get(comb)
            results.loc[tuple(comb), res.index] = res
            res_flip = res.rename({self.RETURN_NAMES[0]: self.RETURN_NAMES[1],
                                   self.RETURN_NAMES[1]: self.RETURN_NAMES[0]})
            results.loc[tuple(comb[::-1]), res_flip.index] = res_flip
        return results
    
    def get_one_vs_rest(self, feature):
        """
        Computes one vs rest decomposition for a given feature
        """
        rest = [col for col in self.fs if col != feature]
        res = self.get([feature, rest])
        return res
    
    def get_all_one_vs_rest(self):
        """
        Computes one vs rest decomposition for all features
        """
        results = pd.DataFrame(index=self.fs, columns=self.RETURN_NAMES)
        for feature in tqdm.tqdm(self.fs):
            results.loc[feature] = self.get_one_vs_rest(feature)
        return results
    
    def get_pairs_vs_rest(self, fixed_feature):
        """
        For a fixed feature, computes pairwise decompositions conditional on the
        respective remainder.
        """
        rest = [f for f in self.fs if f != fixed_feature]
        results = pd.DataFrame(index=rest, columns=self.RETURN_NAMES)
        for feature in tqdm.tqdm(rest):
            C = [f for f in rest if f != feature]
            results.loc[feature] = self.get([[fixed_feature], [feature]], C=C)
        return results    
    
    def hbarplot_comb(self, comb, C=[], ax=None, figsize=None, text=True):
        comb = self.__assert_comb_valid(comb)
        if ax is None:
            f, ax = plt.subplots(figsize=figsize)
        with sns.axes_style('whitegrid'):
            d = self.get(comb, C=C)
            d.plot(kind='barh', ax=ax, xlabel=None, ylabel=None, use_index=False)
            plt.title(f'{comb}')
            sns.despine(left=True, bottom=True, ax=ax)
            return ax
        
    def pairplot(self, figsize=(30, 30), fs=None):
        # Create the grid of subplots
        if fs is None:
            fs = self.fs
        num_features = len(fs)
        fig, axes = plt.subplots(num_features, num_features, figsize=figsize)

        # Iterate over all combinations of the features
        for i, feature_x in enumerate(fs):
            for j, feature_y in enumerate(fs):
                if i != j:
                    ax = axes[i, j]
                    self.hbarplot([feature_x, feature_y], ax=ax, text=False)
        # Adjust layout
        plt.tight_layout()
        return fig, axes
    
    def forceplot_onefixed(self, feature, figsize=None, ax=None, split_additive=False):
        res = self.get_all_pairwise_onefixed(feature)
        data = res.loc[idx[feature, :], :].reset_index().drop('feature1', axis=1).set_index('feature2').transpose()
        ax = forceplot(data, feature, figsize=figsize, ax=ax, split_additive=split_additive)
        return ax

    def forceplots_onefixed(self, figsize=(20, 10), split_additive=False, nrows=1, savepath=None):
        nplots = math.ceil(len(self.fs) / nrows)
        axss = []
        for i in range(nplots):
            # create a figure with #features subplots
            fig, axs = plt.subplots(nrows, 1, figsize=figsize)
            if nrows == 1:
                axs = [axs]
            else:
                axs = axs.flatten()
            
            fs = self.fs[i*nrows:(i+1)*nrows]
            for feature, ax in tqdm.tqdm(zip(fs, axs)):
                # handle logging
                class_level = logging.getLogger('CollabExplainer').getEffectiveLevel()
                logging.getLogger('CollabExplainer').setLevel(logging.WARNING)
                # call plot method
                self.forceplot_onefixed(feature, ax=ax, split_additive=split_additive)
                # handle logging
                logging.getLogger('CollabExplainer').setLevel(class_level)
            plt.tight_layout()
            if savepath is not None:
                plt.savefig(savepath + f'forceplts_{fs}.pdf')
            axss.append(axs)
        return axss
    
    def forceplot_one_vs_rest(self, figsize=(20, 10), split_additive=False, savepath=None):
        res = self.get_all_one_vs_rest()
        data = res.transpose()
        ax = forceplot(data, 'one_vs_rest', figsize=figsize, split_additive=split_additive,
                       explain_surplus=True, rest_feature=2)
        if savepath is not None:
            plt.savefig(savepath + f'forceplt_one_vs_rest.pdf')
        return ax
    
    def forceplot_pairs_vs_rest(self, fixed_feature, figsize=(20, 10), split_additive=False, savepath=None):
        res = self.get_pairs_vs_rest(fixed_feature)
        data = res.transpose()
        ax = forceplot(data, f'{fixed_feature} vs j | rest', figsize=figsize, split_additive=split_additive,
                       explain_collab=True)
        if savepath is not None:
            plt.savefig(savepath + f'forceplt_pairs_vs_rest.pdf')
        return ax
    
    def matrixplots(self, savepath=None):
        tpl = self.get_all_pairwise(return_matrixs=True)
        vars_bivarivate, additive_collab, synergetic_collab, neg2_cov_g1_g2, _ = tpl
        
        cmap = sns.diverging_palette(250, 10, s=80, l=55, as_cmap=True)
        
        fig, axs = plt.subplots(2, 2, figsize=(30, 20))
        sns.heatmap(vars_bivarivate, annot=True, ax=axs[0, 0], vmin=-1, vmax=1, center=0, cmap=cmap)
        axs[0, 0].set_title('Bivariate variance')
        sns.heatmap(additive_collab, annot=True, ax=axs[1, 0], vmin=-1, vmax=1, center=0, cmap=cmap)
        axs[1, 0].set_title('Additive Collaboration')
        sns.heatmap(synergetic_collab, annot=True, ax=axs[1, 1], vmin=-1, vmax=1, center=0, cmap=cmap)
        axs[1, 1].set_title('Interactive Collaboration')
        sns.heatmap(neg2_cov_g1_g2, annot=True, ax=axs[0, 1], vmin=-1, vmax=1, center=0, cmap=cmap)
        axs[0, 1].set_title('Negative Covariance')
        # sns.heatmap(additive_collab_wo_cov, annot=True, ax=axs[1, 2])
        # axs[1, 2].set_title('Additive Collaboration without Covariance')
        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath + 'matrixplots.pdf')
        return axs
        
    def save(self, filepath):
        results = self.get_all_pairwise(only_precomputed=True)
        results.to_csv(filepath)
