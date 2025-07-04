import pandas as pd
from src.preparation import (
    load_data,
    clean_strings,
    split_features_target,
    get_feature_types,
)
from src.modeling import build_preprocessor, build_model
from src.evaluation import evaluate_model
from src.deployment import save_model
from sklearn.model_selection import train_test_split


def main():
    df = load_data("data/ford.csv")
    df = clean_strings(df)
    X, y = split_features_target(df)
    print("Trainingsparameter / Feature-Namen:", list(X.columns))
    numerical_features, categorical_features = get_feature_types(X)
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    model = build_model(preprocessor)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(X_train)
    model.fit(X_train, y_train)
    rmse, r2, y_pred = evaluate_model(model, X_test, y_test)
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R² Score: {r2}")
    # Plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        color="red",
        linestyle="--",
    )
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("Actual vs Predicted Car price")
    plt.show()
    print("Making predictions for the following 5 cars:")
    print(X.head())
    print("The predictions are")
    print(model.predict(X.head()))
    # Save predictions
    results_df = pd.DataFrame({"Echter Wert": y_test, "Vorhergesagt": y_pred})
    results_df["Abweichung"] = results_df["Echter Wert"] - results_df["Vorhergesagt"]
    print(results_df.head(10))
    results_df.to_csv("vorhersagen_knn.csv", index=False)
    save_model(model, "data/ford_model.joblib")


if __name__ == "__main__":
    main()

