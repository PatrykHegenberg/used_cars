import unittest
import pandas as pd
from src.preparation import clean_strings, split_features_target, get_feature_types
from src.modeling import build_preprocessor, build_model
from src.evaluation import evaluate_model

class TestIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        df = pd.DataFrame({
            'year': [2015,2016,2017,2018],
            'model': ['Fiesta','Focus','Fiesta','Focus'],
            'price': [10000, 12000, 11000, 13000]
        })
        df = clean_strings(df)
        X, y = split_features_target(df)
        num, cat = get_feature_types(X)
        preprocessor = build_preprocessor(num, cat)
        model = build_model(preprocessor)
        model.fit(X, y)
        rmse, r2, y_pred = evaluate_model(model, X, y)
        self.assertLess(rmse, 20000)

if __name__ == "__main__":
    unittest.main()