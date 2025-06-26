from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor

def build_preprocessor(numerical_features, categorical_features):
    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

def build_model(preprocessor):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", MLPRegressor(hidden_layer_sizes=(100,), max_iter=3000, random_state=42, early_stopping=True, learning_rate_init=0.001))
    ])