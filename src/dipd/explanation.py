from typing import Any

import matplotlib.axes as mpl_axes
import pandas as pd
from dipd.plots import forceplot

class Explanation:
    def __init__(self, title: str, scores: pd.DataFrame) -> None:
        self.title = title
        self.scores = scores
        self.plot_kwargs: dict[str, Any] = {}

    def __str__(self) -> str:
        return self.title

    def __repr__(self) -> str:
        return self.title

class FeaturewiseExplanation(Explanation):
    """
    Explanations where the result is a dataframe where each row is a feature
    and each column is a scores.
    """
    def __init__(self, title: str, scores: pd.DataFrame) -> None:
        super().__init__(title, scores)

    def forceplot(self, savepath: str | None = None, split_additive: bool = False,
                  figsize: tuple[float, float] = (10, 5),
                  **kwargs: Any) -> mpl_axes.Axes:
        # if a key is both in kwargs and self.plot_kwargs, the value in kwargs is used
        kwargs_pass = {**self.plot_kwargs, **kwargs}
        data = self.scores.transpose()
        ax = forceplot(data, self.title, split_additive=split_additive,
                       figsize=figsize, **kwargs_pass)
        if savepath is not None:
            ax.get_figure().savefig(savepath + self.title + '.pdf')
        return ax

class SurplusExplanation(FeaturewiseExplanation):
    """
    Featurewise explanation where the goal is to explain a surplus over a baseline.
    Thus slightly different hyperparameters for plotting are used.
    """
    def __init__(self, title: str, scores: pd.DataFrame) -> None:
        super().__init__(title, scores)
        self.plot_kwargs = {'explain_surplus': True, 'rest_feature': 2}


class OneFixedExplanation(FeaturewiseExplanation):
    """
    A featurewise explanation with a fixed reference point.
    E.g. all pairs involving feature x1. So for row
    with feature 'x2' the result refers to the pair (x1, x2).
    Potentiall conditional on something else.
    """
    def __init__(self, title: str, scores: pd.DataFrame, fixed_feature: str) -> None:
        super().__init__(title, scores)
        self.fixed_feature = fixed_feature

class CollabExplanation(OneFixedExplanation):
    """
    Slightly different hyperparameters for plotting since we
    are only interested in explaining the collaboration.
    """
    def __init__(self, title: str, scores: pd.DataFrame, fixed_feature: str) -> None:
        super().__init__(title, scores, fixed_feature)
        self.plot_kwargs = {'explain_collab': True}