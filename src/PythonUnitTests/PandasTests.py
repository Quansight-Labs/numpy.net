import unittest
import numpy as np
import pandas as pd  
from nptest import nptest


class Test_PandasTests(unittest.TestCase):


    def test_Pandas_1(self):

        data = pd.DataFrame({"x1":["y", "x", "y", "x", "x", "y"],  # Construct a pandas DataFrame
                     "x22":range(16, 22),
                     "x3":range(1, 7),
                     "x4":["a", "b", "c", "d", "e", "f"],
                     "x5":range(30, 24, - 1)})
        print(data)    
        
        data_row = data[data.x22 < 20]                              # Remove particular rows
        print(data_row)          


    def test_Pandas_2(self):
 
        df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
                   'B': {0: 1, 1: 3, 2: 5},
                   'C': {0: 2, 1: 4, 2: 6}})

        print(df)

 

if __name__ == '__main__':
    unittest.main()
