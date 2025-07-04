import unittest
import numpy as np
from sklearn.dummy import DummyRegressor
from src.evaluation import evaluate_model


class TestEvaluation(unittest.TestCase):
    def test_evaluate_model(self):
        X = np.array([[1], [2], [3]])
        y = np.array([1, 2, 3])
        reg = DummyRegressor(strategy="mean")
        reg.fit(X, y)
        rmse, r2, y_pred = evaluate_model(reg, X, y)
        self.assertIsInstance(rmse, float)
        self.assertIsInstance(r2, float)
        self.assertEqual(len(y_pred), 3)


if __name__ == "__main__":
    unittest.main()

