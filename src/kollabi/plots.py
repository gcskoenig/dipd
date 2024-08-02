import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import numpy as np
import pandas as pd

idx = pd.IndexSlice

FORCEPLOT_COLOR_DICT = {'additive_collab': '#9F76F5', 'additive_collab_explv': '#C776F5',
                        'additive_collab_cov': '#7677F5', 'interactive_collab': '#7AF58D',
                        'var_g2': 'gray', 'var_g1': 'darkgrey', 'total': 'black',
                        'var_gC': 'lightgrey'}
    

def forceplot(data, title_fs_name, figsize=None, ax=None, split_additive=False, color_dict=None,
              explain_surplus=False, rest_feature=2, explain_collab=False):
    """
    Forecplot that takes the results of decompositions and plots them as a stacked bar plot.
    data: pd.DataFrame with the decomposition scores as index and the columns as the features
    """
    assert not (explain_collab and explain_surplus), 'Cannot explain both collab and surplus'
    
    BAR_WIDTH = 0.6  # determines width of the bars
    HLINE_WIDTH = 0.6
    SEPARATOR_IDENT_PROP = 0.03
    if split_additive:
        BAR_WIDTH = BAR_WIDTH / 2

    # determines colors and order of the stacked bars
    COLOR_DICT = color_dict
    if COLOR_DICT is None:
        COLOR_DICT = FORCEPLOT_COLOR_DICT
    patches = [mpatches.Patch(color=color, label=label) for label, color in COLOR_DICT.items()]        

    data.loc['additive_collab'] = data.loc['additive_collab_cov'] + data.loc['additive_collab_explv']
    split_scores = ['additive_collab_cov', 'additive_collab_explv']
    if not split_additive:
        data.drop(split_scores, axis=0, inplace=True)
    
    # sort scores according to COLOR_DICT
    data = data.loc[sorted(data.index, key=tuple(COLOR_DICT.keys()).index)]
    # data = res
    
    # sort features by total score
    normal_scores = [col for col in list(data.index) if col not in split_scores] # names of the decomposition score except additive split scores
    if explain_surplus:
        normal_scores = [col for col in normal_scores if f'{rest_feature}' not in col]
    if explain_collab:
        normal_scores = [col for col in normal_scores if 'collab' in col]
    total_scores = data.loc[normal_scores, :].sum(axis=0)
    data = data[total_scores.sort_values(ascending=False).index]
    total_scores = total_scores[data.columns]
    
    # take total score, add negative values to get max, subtract positive values to get min
    max_val = (total_scores + data.loc[normal_scores, :].where(data.loc[normal_scores, :] <= 0, 0).abs().sum(axis=0)).max()
    min_val = (total_scores - data.loc[normal_scores, :].where(data.loc[normal_scores, :] > 0, 0).sum(axis=0)).min()
    
    if split_additive:
        max_val_split = (total_scores + data.loc[split_scores,:].where(data.loc[split_scores,:] <= 0, 0).abs().sum(axis=0)).max()
        min_val_split = (total_scores - data.loc[split_scores,:].where(data.loc[split_scores,:] > 0, 0).sum(axis=0)).min()
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
        ax.axhline(0, color='black', linewidth=1)


        # Base position for the bars
        bar_positions = np.arange(len(feature_names))
        ax.hlines(total_scores[feature_names], bar_positions - HLINE_WIDTH/2 - DELTA_X, bar_positions + HLINE_WIDTH/2 - DELTA_X, color=COLOR_DICT['total'], linewidth=2)

        # Initialize the bottom arrays for stacking
        positive_top = np.array(total_scores[feature_names]) - SEPARATOR_IDENT
        positive_top_split = np.array(data.loc['additive_collab',feature_names]) - SEPARATOR_IDENT/2
        negative_bottom = np.array(positive_top) + 2*SEPARATOR_IDENT
        negative_bottom_split = np.array(positive_top_split) + SEPARATOR_IDENT
        
        if split_additive:
            ax.hlines(data.loc['additive_collab',feature_names], bar_positions - HLINE_WIDTH/2 + DELTA_X, bar_positions + HLINE_WIDTH/2 + DELTA_X, color=COLOR_DICT['additive_collab'], linewidth=2)

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
                # ax.bar(bar_positions - DELTA_X, positive_values, bottom=positive_bottom, label=label, color=COLOR_DICT[label], width=BAR_WIDTH)
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
                # ax.bar(bar_positions - DELTA_X, -1 * negative_values, bottom=negative_bottom, color=COLOR_DICT[label], width=BAR_WIDTH)
                negative_bottom -= negative_values  # Update the negative bottom for the next stack
                
                # # ADD SEPARATORS
                # for jj in range(len(bar_positions)):
                #     points_separator = [[bar_positions[jj] - DELTA_X - BAR_WIDTH/2, positive_bottom[jj] - SEPARATOR_IDENT],
                #                         [bar_positions[jj] - DELTA_X, positive_bottom[jj]],
                #                         [bar_positions[jj] - DELTA_X + BAR_WIDTH/2, positive_bottom[jj] - SEPARATOR_IDENT]]

                #     line = plt.Polygon(points_separator, closed=None, fill=None,
                #                        edgecolor=COLOR_DICT[label], lw=3)
                #     ax.add_patch(line)
                
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

                # ax.bar(bar_positions + DELTA_X, positive_values, bottom=positive_bottom_split, label=label, color=COLOR_DICT[label], width=BAR_WIDTH/2)
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

                # ax.bar(bar_positions + DELTA_X, -1 * negative_values, bottom=negative_bottom_split, color=COLOR_DICT[label], width=BAR_WIDTH/2)
                negative_bottom_split -= negative_values                
                            

        # Set the x-axis labels
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(feature_names)
        sns.despine(top=True, right=True, left=True, bottom=True, ax=ax)

        # Add legend
        ax.legend()
        
        # Add labels and title
        ax.set_ylabel('Proportion of Var(Y)')
        ax.set_title('Collab explanations for feature {} (f1)'.format(title_fs_name))
        margin = 0.05 * (max_val - min_val)
        ax.set_ylim(min_val - margin, max_val + margin)
        plt.legend(handles=patches, loc='upper right')
        
        return ax        
