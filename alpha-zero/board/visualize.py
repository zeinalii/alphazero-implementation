#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.table import Table
import pandas as pd
import numpy as np


def view_board(np_data, fmt='{:s}', bkg_colors=None):
    if bkg_colors is None:
        bkg_colors = ['yellow', 'white']
        
    # Create a DataFrame with specified column headers
    dataframe = pd.DataFrame(np_data, columns=['A','B','C','D','E','F','G','H'])
    
    # Initialize the plot
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.axis('off')  # Hide the axes
    
    # Create a table within the plot
    table = Table(axis, bbox=[0, 0, 1, 1])
    num_rows, num_cols = dataframe.shape
    cell_width = 1.0 / num_cols
    cell_height = 1.0 / num_rows

    # Populate the table cells with data and background colors
    for (row, col), value in np.ndenumerate(dataframe.values):
        color_index = (row + col) % 2
        background = bkg_colors[color_index]
        table.add_cell(row, col, cell_width, cell_height, text=fmt.format(value),
                      loc='center', facecolor=background)

    # Add row labels
    for row_idx, row_label in enumerate(dataframe.index):
        table.add_cell(row_idx, -1, cell_width, cell_height, text=row_label,
                      loc='right', edgecolor='none', facecolor='none')

    # Add column headers
    for col_idx, col_label in enumerate(dataframe.columns):
        table.add_cell(-1, col_idx, cell_width, cell_height / 2, text=col_label,
                      loc='center', edgecolor='none', facecolor='none')

    # Set the font size and add the table to the plot
    table.set_fontsize(24)
    axis.add_table(table)
    
    return figure
