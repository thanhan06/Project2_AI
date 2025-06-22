"""
Model training and evaluation utilities for Decision Tree Project
CS14003 - Introduction to Artificial Intelligence
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def train_decision_trees(splits, criterion='gini', random_state=42):
    """
    Train decision tree classifiers for all train-test splits
    
    Args:
        splits: Dictionary containing train-test splits
        criterion: Split criterion ('gini' or 'entropy')
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary containing trained models and predictions
    """
    results = {}
    
    for split_name, split_data in splits.items():
        # Train decision tree
        dt = DecisionTreeClassifier(
            criterion=criterion,
            random_state=random_state
        )
        
        dt.fit(split_data['X_train'], split_data['y_train'])
        
        # Make predictions
        y_train_pred = dt.predict(split_data['X_train'])
        y_test_pred = dt.predict(split_data['X_test'])
        
        # Calculate accuracies
        train_accuracy = accuracy_score(split_data['y_train'], y_train_pred)
        test_accuracy = accuracy_score(split_data['y_test'], y_test_pred)
        
        results[split_name] = {
            'model': dt,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'X_train': split_data['X_train'],
            'X_test': split_data['X_test'],
            'y_train': split_data['y_train'],
            'y_test': split_data['y_test']
        }
    
    return results

def evaluate_models(results, dataset_name="Dataset"):
    """
    Generate comprehensive evaluation reports for all models
    
    Args:
        results: Dictionary containing model results
        dataset_name: Name of dataset for reporting
    """
    print(f"\n{dataset_name} - Model Evaluation Results")
    print("="*60)
    
    # Summary table
    summary_data = []
    for split_name, result in results.items():
        summary_data.append({
            'Split': split_name,
            'Train Accuracy': f"{result['train_accuracy']:.4f}",
            'Test Accuracy': f"{result['test_accuracy']:.4f}",
            'Train Samples': len(result['y_train']),
            'Test Samples': len(result['y_test'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nAccuracy Summary:")
    print(summary_df.to_string(index=False))
    
    # Detailed reports for each split
    for split_name, result in results.items():
        print(f"\n{'-'*40}")
        print(f"DETAILED EVALUATION - {split_name} Split")
        print(f"{'-'*40}")
        
        # Classification report
        print(f"\nClassification Report (Test Set):")
        print(classification_report(result['y_test'], result['y_test_pred']))
        
        # Confusion matrix
        print(f"\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(result['y_test'], result['y_test_pred'])
        print(cm)

def plot_confusion_matrices(results, dataset_name="Dataset"):
    """
    Plot confusion matrices for all models
    
    Args:
        results: Dictionary containing model results
        dataset_name: Name of dataset for plot titles
    """
    n_models = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (split_name, result) in enumerate(results.items()):
        if idx >= 4:  # Only plot first 4 splits
            break
            
        cm = confusion_matrix(result['y_test'], result['y_test_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{dataset_name}\n{split_name} Split')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    # Remove unused subplots
    for idx in range(len(results), 4):
        axes[idx].remove()
    
    plt.tight_layout()
    plt.show()

def depth_analysis(X_train, X_test, y_train, y_test, max_depths=[None, 2, 3, 4, 5, 6, 7], 
                  criterion='gini', random_state=42):
    """
    Analyze the effect of tree depth on accuracy
    
    Args:
        X_train, X_test, y_train, y_test: Train-test split data
        max_depths: List of max_depth values to test
        criterion: Split criterion
        random_state: Random seed
    
    Returns:
        Dictionary containing depth analysis results
    """
    depth_results = {}
    
    for max_depth in max_depths:
        # Train model with specified depth
        dt = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_state
        )
        
        dt.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = dt.predict(X_train)
        y_test_pred = dt.predict(X_test)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        depth_key = str(max_depth) if max_depth is not None else 'None'
        depth_results[depth_key] = {
            'model': dt,
            'max_depth': max_depth,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
    
    return depth_results

def plot_depth_analysis(depth_results, dataset_name="Dataset"):
    """
    Plot depth vs accuracy analysis
    
    Args:
        depth_results: Results from depth_analysis function
        dataset_name: Name of dataset for plot title
    """
    # Prepare data for plotting
    depths = []
    train_accuracies = []
    test_accuracies = []
    
    for depth_key, result in depth_results.items():
        if depth_key == 'None':
            depths.append('None')
        else:
            depths.append(int(depth_key))
        train_accuracies.append(result['train_accuracy'])
        test_accuracies.append(result['test_accuracy'])
    
    # Create depth vs accuracy plot
    plt.figure(figsize=(10, 6))
    
    x_pos = range(len(depths))
    plt.plot(x_pos, train_accuracies, 'o-', label='Train Accuracy', linewidth=2, markersize=8)
    plt.plot(x_pos, test_accuracies, 's-', label='Test Accuracy', linewidth=2, markersize=8)
    
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title(f'{dataset_name} - Decision Tree Depth vs Accuracy')
    plt.xticks(x_pos, depths)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    # Add value labels on points
    for i, (train_acc, test_acc) in enumerate(zip(train_accuracies, test_accuracies)):
        plt.annotate(f'{train_acc:.3f}', (i, train_acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
        plt.annotate(f'{test_acc:.3f}', (i, test_acc), textcoords="offset points", 
                    xytext=(0,-15), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print accuracy table
    print(f"\n{dataset_name} - Depth vs Accuracy Table:")
    print("="*50)
    
    depth_df = pd.DataFrame({
        'max_depth': depths,
        'Train Accuracy': [f"{acc:.4f}" for acc in train_accuracies],
        'Test Accuracy': [f"{acc:.4f}" for acc in test_accuracies]
    })
    
    print(depth_df.to_string(index=False))

def get_feature_importance(model, feature_names):
    """
    Get and display feature importance from trained decision tree
    
    Args:
        model: Trained DecisionTreeClassifier
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importance
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df

def plot_feature_importance(model, feature_names, dataset_name="Dataset", top_n=10):
    """
    Plot feature importance from decision tree
    
    Args:
        model: Trained DecisionTreeClassifier
        feature_names: List of feature names
        dataset_name: Name of dataset for plot title
        top_n: Number of top features to display
    """
    importance_df = get_feature_importance(model, feature_names)
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title(f'{dataset_name} - Top {top_n} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
    
    return importance_df