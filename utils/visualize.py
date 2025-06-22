"""
Visualization utilities for Decision Tree Project
CS14003 - Introduction to Artificial Intelligence
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree, export_graphviz
import pandas as pd
import numpy as np
from typing import List, Dict
import io
import pydotplus
from IPython.display import Image

def visualize_decision_tree(model, feature_names, class_names, max_depth=3, 
                          figsize=(20, 10), dataset_name="Dataset"):
    """
    Visualize decision tree using matplotlib
    
    Args:
        model: Trained DecisionTreeClassifier
        feature_names: List of feature names
        class_names: List of class names
        max_depth: Maximum depth to display
        figsize: Figure size
        dataset_name: Dataset name for title
    """
    plt.figure(figsize=figsize)
    
    plot_tree(model, 
             feature_names=feature_names,
             class_names=[str(cls) for cls in class_names],
             filled=True,
             rounded=True,
             max_depth=max_depth,
             fontsize=10)
    
    actual_depth = model.get_depth()
    display_depth = min(max_depth, actual_depth) if max_depth else actual_depth
    
    plt.title(f'{dataset_name} - Decision Tree (Depth: {display_depth})', fontsize=16)
    plt.tight_layout()
    plt.show()

def export_tree_graphviz(model, feature_names, class_names, filename=None):
    """
    Export decision tree to Graphviz format
    
    Args:
        model: Trained DecisionTreeClassifier
        feature_names: List of feature names
        class_names: List of class names
        filename: Output filename (optional)
    
    Returns:
        Graphviz DOT data
    """
    dot_data = export_graphviz(model,
                              out_file=None,
                              feature_names=feature_names,
                              class_names=[str(cls) for cls in class_names],
                              filled=True,
                              rounded=True,
                              special_characters=True)
    
    if filename:
        with open(filename, 'w') as f:
            f.write(dot_data)
    
    return dot_data

def create_depth_comparison_visualization(depth_results, dataset_name="Dataset"):
    """
    Create comprehensive visualization comparing different tree depths
    
    Args:
        depth_results: Results from depth analysis
        dataset_name: Dataset name for titles
    """
    n_depths = len(depth_results)
    
    # Create subplots for tree visualizations (first few depths only)
    max_trees_to_show = min(4, n_depths)
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))