import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_strings(df):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def split_features_target(df, target="price"):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def get_feature_types(X):
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    return numerical_features, categorical_features