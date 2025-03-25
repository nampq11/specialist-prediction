import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.sparse import issparse
from sklearn.pipeline import Pipeline

def chunk_transform(df: pd.DataFrame, pipeline: Pipeline, chunk_size=1000):
    """Transform a DataFrame using a sklearn pipeline in chunks to avoid memory issues.
    This function processes large DataFrames by splitting them into smaller chunks,
    applying the transformation pipeline to each chunk, and then combining the results.
    Handles both dense and sparse outputs from the pipeline.
        Input DataFrame to transform
    pipeline : Pipeline
        Scikit-learn Pipeline object with a transform method
    chunk_size : int, default=1000
        Size of chunks to process at a time
    np.ndarray
        Transformed features as a numpy array
    Notes:
    ------
    - Displays a progress bar using tqdm
    - Automatically converts sparse matrices to dense arrays
    - Combines all transformed chunks into a single array using np.vstack
    """
    transformed_chunks = []

    progress_bar = tqdm(range(0, df.shape[0], chunk_size), desc="Transforming chunks")

    for start in progress_bar:
        end = min(start + chunk_size, df.shape[0])
        chunk_df = df.iloc[start:end]

        transformed_chunk = pipeline.transform(chunk_df)

        if issparse(transformed_chunk):
            transformed_chunk = transformed_chunk.toarray()
        
        transformed_chunks.append(transformed_chunk)
    
    transformed_full = np.vstack(transformed_chunks)

    return transformed_full

def add_transformed_feature(df: pd.DataFrame, transformed_features: np.ndarray, 
                           feature_idx: int = 0, feature_name: str = None):
    """
    Add a single transformed feature as a new column to the original DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original DataFrame to which the feature will be added
    transformed_features : np.ndarray
        Array of transformed features (1D or 2D)
    feature_idx : int, default=0
        Index of the feature to add from transformed_features
    feature_name : str, default=None
        Name for the feature column. If None, uses "feature_{feature_idx}"
        
    Returns:
    --------
    pd.DataFrame
        Original DataFrame with the new feature column added
    """
    result_df = df.copy()
    
    # Handle 1D arrays
    if len(transformed_features.shape) == 1:
        values = transformed_features
    else:
        # Extract the specified column from 2D array
        values = transformed_features[:, feature_idx]
    
    # Use provided feature name or generate one
    column_name = feature_name if feature_name else f"feature_{feature_idx}"
    result_df[column_name] = values
    
    return result_df
