import unittest
import numpy as np
import pandas as pd

from src.ml_tools import one_hot


class TestClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.input_str = ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']
        cls.input_int = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        cls.one_hot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], 
                                [0, 0, 1]])
        return super().setUpClass()
    
    def test_categorical_encode(self):
        # Test for lists, dataframe, and numpy arrays
        # Test str and int
        for value in [self.input_str, self.input_int, 
                      pd.DataFrame(self.input_str)[0], pd.DataFrame(self.input_int)[0], 
                      np.array(self.input_str), np.array(self.input_int)]:
            self.assertTrue(np.array_equal(
                one_hot.categorical_encode(value), 
                self.one_hot
            ))
            print(value)
        
        
if __name__ == '__main__':
    unittest.main()
