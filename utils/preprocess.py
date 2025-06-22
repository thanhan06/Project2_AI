"""
Data preprocessing utilities for Decision Tree Project
CS14003 - Introduction to Artificial Intelligence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from typing import Tuple, List, Dict

def prepare_train_test_splits(X, y, test_sizes=[0.6, 0.4, 0.2, 0.1], random_state=42):
    """
    Prepare multiple train-test splits with different proportions
    
    Args:
        X: Features dataframe
        y: Target series
        test_sizes: List of test proportions [0.6, 0.4, 0.2, 0.1] for 40/60, 60/40, 80/20, 90/10
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary containing all splits
    """
    splits = {}
    
    for test_size in test_sizes:
        train_ratio = int((1 - test_size) * 100)
        test_ratio = int(test_size * 100)
        split_name = f"{train_ratio}_{test_ratio}"
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        splits[split_name] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    return splits

def visualize_class_distributions(y_original, splits, dataset_name="Dataset"):
    """
    Visualize class distributions across original and all split datasets
    
    Args:
        y_original: Original target variable
        splits: Dictionary of train-test splits
        dataset_name: Name of dataset for plot title
    """
    # Calculate number of subplots needed
    n_splits = len(splits)
    fig, axes = plt.subplots(2, n_splits + 1, figsize=(4 * (n_splits + 1), 8))
    
    if n_splits == 0:
        return
    
    # Original distribution
    y_original.value_counts().plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title(f'{dataset_name}\nOriginal Distribution')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    total = len(y_original)
    for i, v in enumerate(y_original.value_counts()):
        axes[0, 0].text(i, v + total*0.01, f'{v}\n({v/total:.1%})', ha='center')
    
    # Remove empty subplot in bottom row
    axes[1, 0].remove()
    
    # Training and test distributions for each split
    for idx, (split_name, split_data) in enumerate(splits.items()):
        col_idx = idx + 1
        
        # Training distribution
        train_counts = split_data['y_train'].value_counts()
        train_counts.plot(kind='bar', ax=axes[0, col_idx], color='lightgreen')
        axes[0, col_idx].set_title(f'{split_name} Split\nTrain Distribution')
        axes[0, col_idx].set_ylabel('Count')
        axes[0, col_idx].tick_params(axis='x', rotation=45)
        
        # Add percentage labels for training
        train_total = len(split_data['y_train'])
        for i, v in enumerate(train_counts):
            axes[0, col_idx].text(i, v + train_total*0.01, f'{v}\n({v/train_total:.1%})', ha='center')
        
        # Test distribution
        test_counts = split_data['y_test'].value_counts()
        test_counts.plot(kind='bar', ax=axes[1, col_idx], color='lightcoral')
        axes[1, col_idx].set_title(f'Test Distribution')
        axes[1, col_idx].set_ylabel('Count')
        axes[1, col_idx].tick_params(axis='x', rotation=45)
        
        # Add percentage labels for test
        test_total = len(split_data['y_test'])
        for i, v in enumerate(test_counts):
            axes[1, col_idx].text(i, v + test_total*0.01, f'{v}\n({v/test_total:.1%})', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{dataset_name} - Class Distribution Summary:")
    print("="*50)
    print(f"Original dataset: {len(y_original)} samples")
    for class_label in y_original.unique():
        count = (y_original == class_label).sum()
        print(f"  Class {class_label}: {count} ({count/len(y_original):.1%})")
    
    print("\nTrain-Test Split Summary:")
    for split_name, split_data in splits.items():
        print(f"\n{split_name} split:")
        print(f"  Train: {split_data['train_size']} samples")
        print(f"  Test: {split_data['test_size']} samples")

def encode_categorical_features(df, categorical_columns, method='onehot'):
    """
    Encode categorical features using specified method
    
    Args:
        df: Input dataframe
        categorical_columns: List of categorical column names
        method: 'onehot' or 'label' encoding
    
    Returns:
        Encoded dataframe and encoders dictionary
    """
    df_encoded = df.copy()
    encoders = {}
    
    for col in categorical_columns:
        if col not in df.columns:
            continue
            
        if method == 'onehot':
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            encoders[col] = dummies.columns.tolist()
            
        elif method == 'label':
            # Label encoding
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    return df_encoded, encoders

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in dataset
    
    Args:
        df: Input dataframe
        strategy: 'mean', 'median', 'mode', or 'drop'
    
    Returns:
        DataFrame with handled missing values
    """
    df_clean = df.copy()
    
    if strategy == 'drop':
        df_clean = df_clean.dropna()
    else:
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['int64', 'float64']:
                    if strategy == 'mean':
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    elif strategy == 'median':
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    # For categorical data, use mode
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    return df_clean

def print_dataset_info(df, dataset_name="Dataset"):
    """
    Print comprehensive information about the dataset
    
    Args:
        df: Input dataframe
        dataset_name: Name of the dataset
    """
    print(f"\n{dataset_name} Information:")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic statistics:\n{df.describe()}")
    
    # Check for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"\nCategorical columns: {categorical_cols}")
        for col in categorical_cols:
            print(f"  {col}: {df[col].unique()}")