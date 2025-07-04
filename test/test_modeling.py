import unittest
import pandas as pd
from src.modeling import build_preprocessor, build_model


class TestModelling(unittest.TestCase):
    def test_build_preprocessor_and_model(self):
        X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "cat": ["A", "B", "A", "B"]})
        y = [10, 20, 30, 40]
        num, cat = ["num"], ["cat"]
        preprocessor = build_preprocessor(num, cat)
        self.assertIsNotNone(preprocessor)
        model = build_model(preprocessor)
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(y))


if __name__ == "__main__":
    unittest.main()
