import unittest
import pandas as pd
from src.modeling import build_preprocessor, build_model
from src.deployment import save_model, load_model

class TestDeployment(unittest.TestCase):
    def test_save_and_load_model(self):
        X = pd.DataFrame({'feature': [1, 2, 3, 4], 'model': ['A', 'B', 'A', 'B']})
        y = [2, 4, 6, 8]
        num, cat = ['feature'], ['model']
        preprocessor = build_preprocessor(num, cat)
        model = build_model(preprocessor)
        model.fit(X, y)
        save_model(model, 'test_model.joblib')
        loaded = load_model('test_model.joblib')
        self.assertTrue(hasattr(loaded, 'predict'))
        import os
        os.remove('test_model.joblib')

if __name__ == '__main__':
    unittest.main()