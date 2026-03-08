import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import numpy as np
import pandas as pd

from dipd.consts import FORCEPLOT_COLOR_DICT

idx = pd.IndexSlice

def forceplot(data, title, figsize=None, ax=None, split_additive=False, color_dict=None,
              explain_surplus=False, rest_feature=2, explain_collab=False, xticks=True,
              xticklabel_rotation=45, center_additive_total=False,
              hline_width=1.0, bar_width=0.6, separator_ident_prop=0.05, hline_thickness=1,
              total_color=None, fontsize=7, fontname='Helvetica', ylabel='Normalized Scores',
              sort_by='score'):
    """
    Forecplot that takes the results of decompositions and plots them as a stacked bar plot.
    data: pd.DataFrame with the decomposition scores as index and the columns as the features
    """
    assert not (explain_collab and explain_surplus), 'Cannot explain both collab and surplus'

    with plt.rc_context({'font.family': fontname, 'pdf.fonttype': 42, 'ps.fonttype': 42}):

        TOTAL_COLOR = FORCEPLOT_COLOR_DICT['total']
        if total_color is not None:
            TOTAL_COLOR = total_color

        data = data.copy()

        BAR_WIDTH = bar_width  # determines width of the bars
        HLINE_WIDTH = hline_width # determines width of the horizontal lines
        HLINE_THIKNESS = hline_thickness
        SEPARATOR_IDENT_PROP = separator_ident_prop # determines wie spitz die spitzen sind
        if split_additive:
            BAR_WIDTH = BAR_WIDTH / 2

        # determines colors and order of the stacked bars
        COLOR_DICT = color_dict
        if COLOR_DICT is None:
            COLOR_DICT = FORCEPLOT_COLOR_DICT

        # build legend color dictionary
        patches = []
        labels = []
        labels.append('total')
        labels.append('main_effect_dependencies')
        labels.append('pure_interactions')
        if not explain_collab:
            labels.append('v2')
            if not explain_surplus:
                labels.append('v1')
        if split_additive:
            labels.append('main_effect_cov')
            labels.append('main_effect_cross_predictability')
        patches = [mpatches.Patch(color=COLOR_DICT[label], label=label) for label in labels]

        data.loc['main_effect_dependencies'] = data.loc['main_effect_cov'] + data.loc['main_effect_cross_predictability']
        split_scores = ['main_effect_cov', 'main_effect_cross_predictability']
        if not split_additive:
            data.drop(split_scores, axis=0, inplace=True)

        # sort scores according to COLOR_DICT
        data = data.loc[sorted(data.index, key=tuple(COLOR_DICT.keys()).index)]

        # sort features by total score
        normal_scores = [col for col in list(data.index) if col not in split_scores] # names of the decomposition score except additive split scores
        if explain_surplus:
            normal_scores = [col for col in normal_scores if f'{rest_feature}' not in col]
        if explain_collab:
            normal_scores = [col for col in normal_scores if 'collab' in col]
        total_scores = data.loc[normal_scores, :].sum(axis=0)

        if sort_by == 'score':
            data = data[total_scores.sort_values(ascending=False).index]
        elif sort_by == 'name':
            data = data[sorted(data.columns)]
        else:
            raise ValueError(f'Unknown sort_by value: {sort_by}')
        total_scores = total_scores[data.columns]

        # take total score, add negative values to get max, subtract positive values to get min
        max_val = (total_scores + data.loc[normal_scores, :].where(data.loc[normal_scores, :] <= 0, 0).abs().sum(axis=0)).max()
        min_val = (total_scores - data.loc[normal_scores, :].where(data.loc[normal_scores, :] > 0, 0).sum(axis=0)).min()

        if split_additive:
            if center_additive_total:
                additive_total = total_scores
            else:
                additive_total = data.loc['main_effect_dependencies']
            max_val_split = (additive_total + data.loc[split_scores,:].where(data.loc[split_scores,:] <= 0, 0).abs().sum(axis=0)).max()
            min_val_split = (additive_total - data.loc[split_scores,:].where(data.loc[split_scores,:] > 0, 0).sum(axis=0)).min()
            max_val = max(max_val, max_val_split)
            min_val = min(min_val, min_val_split)

        SEPARATOR_IDENT = (max_val - min_val) * SEPARATOR_IDENT_PROP

        # List of feature names
        feature_names = list(data.columns)

        # Colors for the stacked bars
        labels = list(data.index)

        # Create the bar plot
        if ax is None:
            f, ax = plt.subplots(figsize=figsize)
        with sns.axes_style('whitegrid'):
            DELTA_X = 0
            if split_additive:
                DELTA_X = 0.25
                HLINE_WIDTH = HLINE_WIDTH / 2

            # black line for the zero
            ax.axhline(0, color='black', linewidth=hline_thickness/2)


            # Base position for the bars
            bar_positions = np.arange(len(feature_names))
            ax.hlines(total_scores[feature_names], bar_positions - HLINE_WIDTH/2 - DELTA_X, bar_positions + HLINE_WIDTH/2 - DELTA_X, color=TOTAL_COLOR,
                      linewidth=HLINE_THIKNESS)

            # Initialize the bottom arrays for stacking
            positive_top = np.array(total_scores[feature_names]) - SEPARATOR_IDENT
            positive_top_split = np.array(data.loc['main_effect_dependencies',feature_names]) - SEPARATOR_IDENT/2
            negative_bottom = np.array(positive_top) + 2*SEPARATOR_IDENT
            negative_bottom_split = np.array(positive_top_split) + SEPARATOR_IDENT

            if split_additive:
                if center_additive_total:
                    raise NotImplementedError('center_additive_total not implemented yet')
                ax.hlines(data.loc['main_effect_dependencies',feature_names], bar_positions - HLINE_WIDTH/2 + DELTA_X, bar_positions + HLINE_WIDTH/2 + DELTA_X,
                          color=COLOR_DICT['main_effect_dependencies'], linewidth=HLINE_THIKNESS)

            first_positive = np.ones(len(bar_positions), dtype=bool)
            first_negative = np.ones(len(bar_positions), dtype=bool)
            # Plot each Series' values as a stacked bar
            for label in labels:  # Iterate based on the length of the first Series
                values = np.array([data[feature].loc[label] for feature in feature_names])

                # Separate positive and negative values
                positive_values = np.where(values > 0, values, 0)
                negative_values = np.where(values < 0, values, 0)

                if label in normal_scores:

                    # Plot positive values
                    positive_bottom = positive_top - positive_values
                    for jj in range(len(bar_positions)):
                        points_rectangle = [[bar_positions[jj] - DELTA_X + BAR_WIDTH/2, positive_top[jj]],
                                            [bar_positions[jj] - DELTA_X + BAR_WIDTH/2, positive_bottom[jj]],
                                            [bar_positions[jj] - DELTA_X, positive_bottom[jj] + SEPARATOR_IDENT],
                                            [bar_positions[jj] - DELTA_X - BAR_WIDTH/2, positive_bottom[jj]],
                                            [bar_positions[jj] - DELTA_X - BAR_WIDTH/2, positive_top[jj]],
                                            [bar_positions[jj] - DELTA_X, positive_top[jj] + SEPARATOR_IDENT]]
                        line = plt.Polygon(points_rectangle, closed=True, fill=True,
                                        facecolor=COLOR_DICT[label], linewidth=0)
                        ax.add_patch(line)
                    positive_top = positive_bottom

                    # Plot negative values
                    for jj in range(len(bar_positions)):
                        points_rectangle = [[bar_positions[jj] - DELTA_X - BAR_WIDTH/2, negative_bottom[jj]],
                                            [bar_positions[jj] - DELTA_X - BAR_WIDTH/2, negative_bottom[jj] - negative_values[jj]],
                                            [bar_positions[jj] - DELTA_X, negative_bottom[jj] - negative_values[jj] - SEPARATOR_IDENT],
                                            [bar_positions[jj] - DELTA_X + BAR_WIDTH/2, negative_bottom[jj] - negative_values[jj]],
                                            [bar_positions[jj] - DELTA_X + BAR_WIDTH/2, negative_bottom[jj]],
                                            [bar_positions[jj] - DELTA_X, negative_bottom[jj] - SEPARATOR_IDENT]]
                        line = plt.Polygon(points_rectangle, closed=True, fill=True,
                                        facecolor=COLOR_DICT[label], linewidth=0)
                        ax.add_patch(line)
                    negative_bottom -= negative_values  # Update the negative bottom for the next stack

                elif split_additive and label in split_scores:
                    positive_bottom_split = positive_top_split - positive_values
                    for jj in range(len(bar_positions)):
                        points_rectangle = [[bar_positions[jj] + DELTA_X + BAR_WIDTH/4, positive_top_split[jj]],
                                            [bar_positions[jj] + DELTA_X + BAR_WIDTH/4, positive_bottom_split[jj]],
                                            [bar_positions[jj] + DELTA_X, positive_bottom_split[jj] + SEPARATOR_IDENT/2],
                                            [bar_positions[jj] + DELTA_X - BAR_WIDTH/4, positive_bottom_split[jj]],
                                            [bar_positions[jj] + DELTA_X - BAR_WIDTH/4, positive_top_split[jj]],
                                            [bar_positions[jj] + DELTA_X, positive_top_split[jj] + SEPARATOR_IDENT/2]]
                        line = plt.Polygon(points_rectangle, closed=True, fill=True,
                                        facecolor=COLOR_DICT[label], linewidth=0)
                        ax.add_patch(line)

                    positive_top_split = positive_bottom_split

                    # Plot negative values
                    for jj in range(len(bar_positions)):
                        points_rectangle = [[bar_positions[jj] + DELTA_X - BAR_WIDTH/4, negative_bottom_split[jj]],
                                            [bar_positions[jj] + DELTA_X - BAR_WIDTH/4, negative_bottom_split[jj] - negative_values[jj]],
                                            [bar_positions[jj] + DELTA_X, negative_bottom_split[jj] - negative_values[jj] - SEPARATOR_IDENT/2],
                                            [bar_positions[jj] + DELTA_X + BAR_WIDTH/4, negative_bottom_split[jj] - negative_values[jj]],
                                            [bar_positions[jj] + DELTA_X + BAR_WIDTH/4, negative_bottom_split[jj]],
                                            [bar_positions[jj] + DELTA_X, negative_bottom_split[jj] - SEPARATOR_IDENT/2]]
                        line = plt.Polygon(points_rectangle, closed=True, fill=True,
                                        facecolor=COLOR_DICT[label], linewidth=0)
                        ax.add_patch(line)

                    negative_bottom_split -= negative_values


            # Set the x-axis labels
            if xticks:
                ax.set_xticks(bar_positions)
                if xticklabel_rotation > 0:
                    ha = 'right'
                else:
                    ha = 'center'
                ax.set_xticklabels(feature_names, rotation=xticklabel_rotation, ha=ha)
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])

            sns.despine(top=True, right=True, left=True, bottom=True, ax=ax)

            # Add labels and title
            margin = 0.05 * (max_val - min_val)
            ax.set_ylim(min_val - margin - SEPARATOR_IDENT, max_val + margin + SEPARATOR_IDENT)
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.set_title(title, fontsize=fontsize)
            legend = ax.legend(handles=patches, loc='upper right', prop={'size': fontsize})
            return ax
