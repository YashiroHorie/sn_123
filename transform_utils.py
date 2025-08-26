import numpy as np
import math
from typing import List, Union, Callable
import logging

logger = logging.getLogger(__name__)

def cubic_transform(embeddings: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Apply cubic transformation: x^3
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        
    Returns:
        Transformed embeddings with cubic transformation applied
    """
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return np.power(embeddings_array, 3)

def sinh_transform(embeddings: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Apply hyperbolic sine transformation: sinh(x)
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        
    Returns:
        Transformed embeddings with sinh transformation applied
    """
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return np.sinh(embeddings_array)

def tanh_transform(embeddings: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Apply hyperbolic tangent transformation: tanh(x)
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        
    Returns:
        Transformed embeddings with tanh transformation applied
    """
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return np.tanh(embeddings_array)

def exp_transform(embeddings: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Apply exponential transformation: e^(x)
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        
    Returns:
        Transformed embeddings with exponential transformation applied
    """
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return np.exp(embeddings_array)

def arctan_transform(embeddings: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Apply arctangent transformation: arctan(x)
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        
    Returns:
        Transformed embeddings with arctan transformation applied
    """
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return np.arctan(embeddings_array)

def square_transform(embeddings: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Apply square transformation: x^2
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        
    Returns:
        Transformed embeddings with square transformation applied
    """
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return np.square(embeddings_array)

def sqrt_transform(embeddings: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Apply square root transformation: sqrt(x)
    Note: Returns NaN for negative values
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        
    Returns:
        Transformed embeddings with square root transformation applied
    """
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return np.sqrt(embeddings_array)

def log_transform(embeddings: Union[List[List[float]], np.ndarray], base: float = math.e) -> np.ndarray:
    """
    Apply logarithmic transformation: log(x) with specified base
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        base: Base for logarithm (default: e)
        
    Returns:
        Transformed embeddings with logarithmic transformation applied
    """
    embeddings_array = np.array(embeddings, dtype=np.float32)
    if base == math.e:
        return np.log(embeddings_array)
    elif base == 2:
        return np.log2(embeddings_array)
    elif base == 10:
        return np.log10(embeddings_array)
    else:
        return np.log(embeddings_array) / np.log(base)

def sigmoid_transform(embeddings: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Apply sigmoid transformation: 1 / (1 + e^(-x))
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        
    Returns:
        Transformed embeddings with sigmoid transformation applied
    """
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return 1 / (1 + np.exp(-embeddings_array))

def relu_transform(embeddings: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Apply ReLU transformation: max(0, x)
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        
    Returns:
        Transformed embeddings with ReLU transformation applied
    """
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return np.maximum(0, embeddings_array)

def softplus_transform(embeddings: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Apply softplus transformation: ln(1 + e^x)
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        
    Returns:
        Transformed embeddings with softplus transformation applied
    """
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return np.log(1 + np.exp(embeddings_array))

def cosine_transform(embeddings: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Apply cosine transformation: cos(x)
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        
    Returns:
        Transformed embeddings with cosine transformation applied
    """
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return np.cos(embeddings_array)

def sine_transform(embeddings: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Apply sine transformation: sin(x)
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        
    Returns:
        Transformed embeddings with sine transformation applied
    """
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return np.sin(embeddings_array)

def abs_transform(embeddings: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Apply absolute value transformation: |x|
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        
    Returns:
        Transformed embeddings with absolute value transformation applied
    """
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return np.abs(embeddings_array)

def normalize_transform(embeddings: Union[List[List[float]], np.ndarray], 
                       method: str = 'minmax') -> np.ndarray:
    """
    Apply normalization transformation
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        method: Normalization method ('minmax', 'zscore', 'l2')
        
    Returns:
        Normalized embeddings
    """
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        min_vals = np.min(embeddings_array, axis=0, keepdims=True)
        max_vals = np.max(embeddings_array, axis=0, keepdims=True)
        return (embeddings_array - min_vals) / (max_vals - min_vals + 1e-8)
    
    elif method == 'zscore':
        # Z-score normalization
        mean_vals = np.mean(embeddings_array, axis=0, keepdims=True)
        std_vals = np.std(embeddings_array, axis=0, keepdims=True)
        return (embeddings_array - mean_vals) / (std_vals + 1e-8)
    
    elif method == 'l2':
        # L2 normalization
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        return embeddings_array / (norms + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def clip_transform(embeddings: Union[List[List[float]], np.ndarray], 
                  min_val: float = -1.0, max_val: float = 1.0) -> np.ndarray:
    """
    Apply clipping transformation to keep values within specified range
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        min_val: Minimum value for clipping
        max_val: Maximum value for clipping
        
    Returns:
        Clipped embeddings
    """
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return np.clip(embeddings_array, min_val, max_val)

def apply_transform_pipeline(embeddings: Union[List[List[float]], np.ndarray], 
                           transforms: List[Callable]) -> np.ndarray:
    """
    Apply a pipeline of transformations sequentially
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        transforms: List of transformation functions to apply
        
    Returns:
        Transformed embeddings after applying all transformations
    """
    result = np.array(embeddings, dtype=np.float32)
    
    for i, transform_func in enumerate(transforms):
        try:
            result = transform_func(result)
            logger.debug(f"Applied transform {i+1}/{len(transforms)}: {transform_func.__name__}")
        except Exception as e:
            logger.error(f"Error applying transform {transform_func.__name__}: {e}")
            raise
    
    return result

def get_available_transforms() -> dict:
    """
    Get a dictionary of all available transformation functions
    
    Returns:
        Dictionary mapping transform names to their functions
    """
    return {
        'cubic': cubic_transform,
        'sinh': sinh_transform,
        'tanh': tanh_transform,
        'exp': exp_transform,
        'arctan': arctan_transform,
        'square': square_transform,
        'sqrt': sqrt_transform,
        'log': log_transform,
        'sigmoid': sigmoid_transform,
        'relu': relu_transform,
        'softplus': softplus_transform,
        'cosine': cosine_transform,
        'sine': sine_transform,
        'abs': abs_transform,
        'normalize': normalize_transform,
        'clip': clip_transform
    }

def transform_embeddings_by_name(embeddings: Union[List[List[float]], np.ndarray], 
                                transform_name: str, **kwargs) -> np.ndarray:
    """
    Apply transformation by name with optional parameters
    
    Args:
        embeddings: Input embeddings (list of lists or numpy array)
        transform_name: Name of the transformation to apply
        **kwargs: Additional arguments for the transformation function
        
    Returns:
        Transformed embeddings
    """
    available_transforms = get_available_transforms()
    
    if transform_name not in available_transforms:
        raise ValueError(f"Unknown transform: {transform_name}. Available: {list(available_transforms.keys())}")
    
    transform_func = available_transforms[transform_name]
    
    # Handle inhomogeneous embeddings (different dimensions per asset)
    if isinstance(embeddings, list) and len(embeddings) > 0:
        # Check if embeddings have different shapes
        shapes = [len(emb) for emb in embeddings]
        if len(set(shapes)) > 1:
            # Process each asset's embeddings separately
            transformed_embeddings = []
            for i, asset_embedding in enumerate(embeddings):
                try:
                    # Convert single asset embedding to numpy array
                    asset_array = np.array(asset_embedding, dtype=np.float32)
                    # Apply transformation to this asset
                    transformed_asset = transform_func(asset_array, **kwargs)
                    # Convert back to list
                    transformed_embeddings.append(transformed_asset.tolist())
                    logger.debug(f"Applied {transform_name} to asset {i} (shape: {asset_array.shape})")
                except Exception as e:
                    logger.error(f"Failed to transform asset {i}: {e}")
                    # Fall back to original embedding for this asset
                    transformed_embeddings.append(asset_embedding)
            return transformed_embeddings
    
    # If homogeneous or single embedding, apply normally
    return transform_func(embeddings, **kwargs)
