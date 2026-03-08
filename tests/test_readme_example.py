"""
Smoke test: runs the example from README.md as closely as possible.
Uses a non-interactive matplotlib backend to avoid blocking on plt.show().
"""
import matplotlib
matplotlib.use('Agg')  # must be set before importing pyplot

import pandas as pd
import category_encoders as ce
import matplotlib.pyplot as plt

from dipd import DIP
from dipd.learners import EBM
from dipd.plots import forceplot


def test_readme_example():
    # --- load and preprocess data (from README) ---
    varnames = ['longitude', 'latitude', 'ocean_proximity', 'median_house_value']
    target_variable = 'median_house_value'
    df = pd.read_csv(
        'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv'
    ).dropna()[varnames]
    encoder = ce.OrdinalEncoder()
    df = encoder.fit_transform(df)

    # --- DIP decomposition (from README) ---
    explainer = DIP(df, target_variable, EBM)
    explanation = explainer.get_all_loo()
    print(explanation.scores)

    # --- plot (from README, plt.show() replaced with savefig to avoid blocking) ---
    ax = forceplot(explanation.scores.T, 'DIP Decomposition of LOCO scores',
                   figsize=(3, 3), explain_surplus=True)
    ax.get_legend().remove()
    plt.savefig('/tmp/test_readme_forceplot.png')
    plt.close()

    # basic sanity checks on the output
    features = ['longitude', 'latitude', 'ocean_proximity']
    assert list(explanation.scores.index) == features, \
        f"Expected features {features}, got {list(explanation.scores.index)}"
    assert not explanation.scores.isnull().any().any(), \
        "Scores contain NaN values"
