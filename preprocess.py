import pandas as pd
import numpy as np
import json
import os
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_json_data(filepath):
    """Load JSON data from specified filepath"""
    with open(filepath, 'r') as json_data:
        data = json.load(json_data)
    return data

def normalize_json_to_df(data):
    """Normalize JSON data into pandas DataFrame, focusing on appraisals"""
    # Check if 'appraisals' is in the top-level keys
    if 'appraisals' in data:
        # Correctly normalize the data by accessing the appraisals list
        df = pd.json_normalize(data['appraisals'])
        return df
    else:
        # If structure is different, try direct normalization
        return pd.json_normalize(data)

# Standardization functions from eda.ipynb
def standardize_gla(value):
    """
    Standardizes the GLA (Gross Living Area) value to float (in SqFt).
    Handles values in SqFt, SqM, sf, and with commas, decimals, or +/-.
    """
    if not isinstance(value, str):
        return None

    # Remove commas and plus/minus
    value = value.replace(',', '').replace('+/-', '').strip()

    # Extract the numeric part
    match = re.match(r"([\d\.]+)", value)
    if not match:
        return None

    num = float(match.group(1))

    # Check for units
    value_lower = value.lower()
    if 'sqm' in value_lower or 'sq m' in value_lower or 'sqmeter' in value_lower or 'sq meter' in value_lower:
        # Convert SqM to SqFt
        num = num * 10.7639
    # If it's already in SqFt or sf, no conversion needed

    return round(num, 2)

def standardize_lot_size(value):
    """
    Standardizes the lot_size_sf value to int (in SqFt).
    Handles values in SqFt, SqM, Acres, and various notations.
    Returns None for missing or non-numeric/condo/NA values.
    """
    if not isinstance(value, str):
        return None

    value = value.lower().replace(',', '').replace('+/-', '').strip()
    # Handle missing/condo/NA
    if any(x in value for x in ['n/a', 'na', 'condo', 'sqft' if value.strip() == 'sqft' else '', 'land']) or value in ['','sqft']:
        return None

    # Extract the numeric part
    match = re.match(r"([\d\.]+)", value)
    if not match:
        return None

    num = float(match.group(1))

    # Check for units
    if 'acre' in value or 'ac' in value:
        num = num * 43560  # 1 acre = 43,560 SqFt
    elif 'sqm' in value or 'sq m' in value:
        num = num * 10.7639  # 1 SqM = 10.7639 SqFt
    # If it's already in SqFt or sqft, no conversion needed

    return int(round(num))

def standardize_year_built(value):
    """
    Standardizes the year_built value to int.
    Handles numeric values and returns None for non-numeric values.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
        
    try:
        if isinstance(value, str):
            # Extract just the first 4-digit number if there's text
            match = re.search(r'\b(19|20)\d{2}\b', value)
            if match:
                return int(match.group(0))
            else:
                return None
        else:
            return int(value)
    except:
        return None

def standardize_effective_age(value):
    """
    Standardizes the effective_age value to int.
    Handles 'New', 'new', and nan.
    Returns 0 for 'New'/'new', None for nan or non-numeric values.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    if isinstance(value, str):
        value = value.strip().lower()
        if value == 'new':
            return 0
        if value.isdigit():
            return int(value)
        else:
            return None
    elif isinstance(value, int):
        return value
    else:
        return None

def standardize_num_beds(value):
    """
    Standardizes the num_beds value to int.
    Handles values like '3', '3+1', '2+2', and nan.
    Sums numbers if '+' is present. Returns None for nan or non-numeric values.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    if isinstance(value, str):
        value = value.strip()
        if '+' in value:
            # Sum all numbers separated by '+'
            numbers = re.findall(r'\d+', value)
            if numbers:
                return sum(int(n) for n in numbers)
            else:
                return None
        elif value.isdigit():
            return int(value)
        else:
            return None
    elif isinstance(value, int):
        return value
    else:
        return None

def standardize_num_baths(value):
    """
    Standardizes the num_baths value to float (full + 0.5*half).
    Handles formats like '2:1', '2F 1H', '2 Full/1Half', '3F', '2', and nan.
    Returns None for nan or unparseable values.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    if isinstance(value, str):
        value = value.strip().lower()
        if value == '':
            return None

        # Format: '2:1' (full:half)
        match_colon = re.match(r'(\d+):(\d+)', value)
        if match_colon:
            full = int(match_colon.group(1))
            half = int(match_colon.group(2))
            return full + 0.5 * half

        # Format: '2F 1H', '1F 1H', '3F 1H'
        match_fh = re.findall(r'(\d+)\s*f', value)
        match_hh = re.findall(r'(\d+)\s*h', value)
        if match_fh or match_hh:
            full = int(match_fh[0]) if match_fh else 0
            half = int(match_hh[0]) if match_hh else 0
            return full + 0.5 * half

        # Format: '2 Full/1Half'
        match_full_half = re.match(r'(\d+)\s*full/?(\d*)half?', value)
        if match_full_half:
            full = int(match_full_half.group(1))
            half = int(match_full_half.group(2)) if match_full_half.group(2) else 0
            return full + 0.5 * half

        # Format: '3F', '4F', '1F'
        match_only_f = re.match(r'(\d+)\s*f$', value)
        if match_only_f:
            return int(match_only_f.group(1))

        # Format: '2' (assume full)
        if value.isdigit():
            return int(value)

        return None

    elif isinstance(value, (int, float)):
        return float(value)
    else:
        return None

def standardize_property_features(df):
    """Apply standardization to key property features"""
    # Create a copy to avoid modifying the original DataFrame
    df_std = df.copy()
    
    # Apply standardization functions to each feature
    if 'subject.gla' in df.columns:
        df_std['subject.gla'] = df['subject.gla'].apply(standardize_gla)
    
    if 'subject.lot_size_sf' in df.columns:
        df_std['subject.lot_size_sf'] = df['subject.lot_size_sf'].apply(standardize_lot_size)
        
    if 'subject.year_built' in df.columns:
        df_std['subject.year_built'] = df['subject.year_built'].apply(standardize_year_built)
        
    if 'subject.effective_age' in df.columns:
        df_std['subject.effective_age'] = df['subject.effective_age'].apply(standardize_effective_age)
        
    if 'subject.num_beds' in df.columns:
        df_std['subject.num_beds'] = df['subject.num_beds'].apply(standardize_num_beds)
        
    if 'subject.num_baths' in df.columns:
        df_std['subject.num_baths'] = df['subject.num_baths'].apply(standardize_num_baths)
    
    return df_std

def extract_numeric_features(df):
    """Extract only numeric features from the DataFrame"""
    # First, identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Also check for potentially numeric columns that might be stored as strings
    for col in df.select_dtypes(include=['object']).columns:
        # Try to convert to numeric, ignoring errors
        try:
            # If more than 70% of values can be converted to numeric, keep the column
            numeric_values = pd.to_numeric(df[col], errors='coerce')
            if numeric_values.notna().mean() > 0.7:  # If >70% are not NaN
                df[col] = numeric_values
                numeric_cols.append(col)
        except:
            continue
    
    # Return DataFrame with only numeric columns
    return df[numeric_cols]

def standardize_features(df, method='standard', return_scaler=False):
    """Standardize numeric features using specified method"""
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")
    
    # Fit and transform the data
    scaled_data = scaler.fit_transform(df)
    
    # Convert back to DataFrame with original column names
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    if return_scaler:
        return scaled_df, scaler
    else:
        return scaled_df

def preprocess_data(json_filepath, method='standard', return_scaler=False, selected_features=None):
    """Main function to preprocess the data for model training"""
    # Load data
    data = load_json_data(json_filepath)
    
    # Normalize to DataFrame
    df = normalize_json_to_df(data)
    
    # Standardize specific property features (before numeric extraction)
    df = standardize_property_features(df)
    
    # Extract all numeric features or use selected features if provided
    if selected_features:
        # Make sure all requested features exist in the dataframe
        available_features = [f for f in selected_features if f in df.columns]
        if len(available_features) < len(selected_features):
            missing = set(selected_features) - set(available_features)
            print(f"Warning: The following requested features were not found: {missing}")
        
        numeric_df = df[available_features].copy()
        # Ensure all values are numeric
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    else:
        numeric_df = extract_numeric_features(df)
    
    # Drop rows with any NaN values
    numeric_df = numeric_df.dropna()
    
    # Standardize (scale) the features
    if return_scaler:
        scaled_df, scaler = standardize_features(numeric_df, method=method, return_scaler=True)
        return scaled_df, scaler
    else:
        scaled_df = standardize_features(numeric_df, method=method)
        return scaled_df

if __name__ == "__main__":
    # Example usage
    # preprocessed_data = preprocess_data("path/to/your/data.json")
    pass
