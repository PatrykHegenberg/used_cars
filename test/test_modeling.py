import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.modeling import build_preprocessor, build_model

import unittest
import pandas as pd


class TestModelling(unittest.TestCase):
    def test_build_preprocessor_and_model(self):
        X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "cat": ["A", "B", "A", "B"]})
        y = [10, 20, 30, 40]
        num, cat = ["num"], ["cat"]
        preprocessor = build_preprocessor(num, cat)
        model = build_model(preprocessor)
        # Setze early_stopping=False für kleine Tests
        model.named_steps["regressor"].early_stopping = False
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(y))


if __name__ == "__main__":
    unittest.main()
