"""
Collection of functions to pre-process/transform data.

@author: Instructor Zhang
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def preprocessor(df, target):
    """
    Pre-process the original data with imputation, normalization
    and digitization.
    
    Parameter:
        df: dataframe, original dataset
        target: string, name of the target
        
    Return:
        Pre-processed dataframe
    """
    X_df = df.drop(labels=[target], axis=1)
    y_df = df[target]
    
    num_features = [
        col for col in X_df.columns if X_df[col].dtype in('int', 'float')
    ]

    # intuitive way to infer categorical feature, can be improved
    cat_features = [col for col in X_df.columns if X_df[col].dtype == 'object']
    
    # create pipeline for numerical and categorical feature, respectively
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # one-hot encode the categorical features
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # stack two transformers
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    
    # transform numerical/categorical feature columns
    if len(cat_features) > 0:
        # if there exists categorical feature
        preprocessor.fit(X_df)
        new_cat_features = (preprocessor.named_transformers_['cat'].
                            named_steps['onehot'].get_feature_names())
        new_features = num_features + list(new_cat_features)
        
        processed_df = preprocessor.transform(X_df)
        processed_df = pd.DataFrame(processed_df, columns=new_features)
    else:
        processed_df = preprocessor.fit_transform(X_df)
        processed_df = pd.DataFrame(processed_df, columns=X_df.columns)
    
    # Combine feature columns with target column
    processed_df.insert(loc=len(processed_df.columns), column=target, value=y_df)
    return processed_df


def splitter(df, target, task_type):
    """Split dataset into training/test with ratio 80%:20%."""
    stratify = df[target] if task_type == 'classification' else None
    
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True, stratify=stratify
    )
    
    return train_df, test_df
