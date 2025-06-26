import unittest
import pandas as pd
from src.preparation import clean_strings, split_features_target, get_feature_types

class TestPreparation(unittest.TestCase):
    def test_clean_strings(self):
        df = pd.DataFrame({'a': [' test ', 'foo', None]})
        df_clean = clean_strings(df)
        self.assertEqual(df_clean['a'][0], 'test')

    def test_split_features_target(self):
        df = pd.DataFrame({'a': [1,2], 'price': [10,20]})
        X, y = split_features_target(df)
        self.assertEqual(list(X.columns), ['a'])
        self.assertTrue((y == pd.Series([10,20], name='price')).all())

    def test_get_feature_types(self):
        df = pd.DataFrame({'a': [1,2], 'b': ['x','y']})
        num, cat = get_feature_types(df)
        self.assertIn('a', num)
        self.assertIn('b', cat)

if __name__ == "__main__":
    unittest.main()