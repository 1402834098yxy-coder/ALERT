#read the json file in this folder
import json
import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
import argparse

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def sort_symmetric_matrix(matrix, indices):
    """
    reorder the matrix by the indices
    Args:
        matrix: original matrix
        indices: label sequence
    
    Returns:
        reordered matrix
    """
    n = len(matrix)
    sorted_indices = indices
    out_matrix = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            index1 = np.where(np.array(sorted_indices) == i)[0]
            index1 = index1[0]
            # index2 = np.where(np.array(sorted_indices) == j)[0]
            # index2 = index2[0]
            out_matrix[i,j] = matrix[index1,j]
    
    return out_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str, default='None')
    parser.add_argument('--output_heatmap_dir', type=str, default='1')
    args = parser.parse_args()
    data = read_json_file(args.input_data_dir)
    figure_path = args.output_heatmap_dir
    probs = data['prob']
    test_probs = probs
    test_matrix = np.array(test_probs)
    test_indices = data['true_label']
    pred_indices = data['pred_label']
    final_indices = data['pred_final_label']

    # get the index of the test_matrix
    pred_indices_get  = np.argmax(test_matrix, axis=1)
    zip_test_indices = list(zip(pred_indices_get,pred_indices))

    unique_sorted_numbers = sorted(set(test_indices))

    mapping = {number: i for i, number in enumerate(unique_sorted_numbers)}

    # Apply this mapping to the original list
    mapped_numbers = [mapping[num] for num in test_indices if num in mapping]
    plot_matrix = sort_symmetric_matrix(test_matrix, mapped_numbers)



    # Example matrix for demonstration purposes
    plot_matrix = plot_matrix[20:30,20:30]  # Replace with your actual matrix






    global_min = min(matrix.min() for matrix in plot_matrix)
    global_max = max(matrix.max() for matrix in plot_matrix)
    norm = Normalize(vmin=0, vmax=1)

    # Create a custom colormap with deeper blue shades
    cmap = cm.Blues
    # norm = Normalize(vmin=np.min(plot_matrix), vmax=)

    # Create the heatmap
    fig, ax = plt.subplots()
    cax = ax.imshow(plot_matrix, cmap=cmap,norm=norm, aspect='auto')

    # Hide the axes
    ax.axis('off')

    # Load Times New Roman font
    font_path = './Times New Roman.ttf'  # Change this path to the location of your Times New Roman font file
    font_prop = FontProperties(fname=font_path, size=16, weight='bold')

    for i in range(plot_matrix.shape[0]):
        for j in range(plot_matrix.shape[1]):
            value = plot_matrix[i, j]
                # Determine text color based on cell value
            text_color = 'white' if value > 0.85 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                    color=text_color, fontproperties=font_prop)
            # ax.text(j, i, f'{plot_matrix[i, j]:.2f}', ha='center', va='center', 
            #         fontproperties=font_prop)

    # Save the figure
    plt.savefig(figure_path, dpi=900, bbox_inches='tight', pad_inches=0)

    # Show the plot
    plt.show()
