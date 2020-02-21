import unittest
import numpy as np


class Test_UFUNC_INT64_Tests(unittest.TestCase):

    #region UFUNC INT64 Tests

    #region OUTER Tests
    def test_AddOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        b = np.add.outer(a1,a2)
        print(b)

    def test_SubtractOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        b = np.subtract.outer(a1,a2)
        print(b)

    def test_MultiplyOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        b = np.multiply.outer(a1,a2)
        print(b)

    def test_MultiplyOuter_INT32(self):

        a1 = np.arange(0, 5, dtype=np.int32);
        a2 = np.arange(3, 8, dtype=np.int32);
  
        b = np.multiply.outer(a1,a2)
        print(b)


    def test_DivideOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        b = np.divide.outer(a1,a2)
        print(b)

    def test_RemainderOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        b = np.remainder.outer(a1,a2)
        print(b)

    def test_FModOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        b = np.fmod.outer(a1,a2)
        print(b)

    def test_SquareOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);

        try :
            b = np.square.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ReciprocalOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        try :
            b = np.reciprocal.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_OnesLikeOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        try :
            b = np.ones_like.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_SqrtOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
        
        try :
            b = np.sqrt.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_NegativeOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        try :
            b = np.negative.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

        
    def test_AbsoluteOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);

        try :
            b = np.absolute.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 

    def test_InvertOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        try :
            b = np.invert.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LeftShiftOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        try :
            b = np.left_shift.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_RightShiftOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        try :
            b = np.right_shift.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

            
    def test_BitwiseAndOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        try :
            b = np.bitwise_and.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

                        
    def test_BitwiseOrOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        try :
            b = np.bitwise_or.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_BitwiseXorOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        try :
            b = np.bitwise_xor.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LessOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        b = np.less.outer(a1,a2)
        print(b)

    def test_LessEqualOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        b = np.less_equal.outer(a1,a2)
        print(b)

    def test_EqualOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        b = np.equal.outer(a1,a2)
        print(b)

    def test_NotEqualOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        b = np.not_equal.outer(a1,a2)
        print(b)

    def test_GreaterOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        b = np.greater.outer(a1,a2)
        print(b)

    def test_GreaterEqualOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        b = np.greater_equal.outer(a1,a2)
        print(b)

    def test_FloorDivideOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        b = np.floor_divide.outer(a1,a2)
        print(b)

    def test_TrueDivideOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        b = np.true_divide.outer(a1,a2)
        print(b)

    def test_LogicalAndOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        b = np.logical_and.outer(a1,a2)
        print(b)

    def test_LogicalOrOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
  
        b = np.logical_or.outer(a1,a2)
        print(b)

    def test_FloorOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);

        try :
            b = np.floor.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_CeilOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);

        try :
            b = np.ceil.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_MaximumOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
 
        b = np.maximum.outer(a1,a2)
        print(b)

    def test_MinimumOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
 
        b = np.minimum.outer(a1,a2)
        print(b)
 
    def test_RintOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);

        try :
            b = np.rint.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ConjugateOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);

        try :
            b = np.conjugate.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_IsNANOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);

        try :
            b = np.isnan.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_FMaxOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
 
        b = np.fmax.outer(a1,a2)
        print(b)

    def test_FMinOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
 
        b = np.fmin.outer(a1,a2)
        print(b)

    def test_HeavisideOuter_INT64(self):

        a1 = np.arange(0, 5, dtype=np.int64);
        a2 = np.arange(3, 8, dtype=np.int64);
 
        b = np.heaviside.outer(a1,a2)
        print(b)
        
      #endregion
  

    #region REDUCE Tests
    def test_AddReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
  
        b = np.add.reduce(a1)
        print(b)

    def test_SubtractReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10)); 
        
        b = np.subtract.reduce(a1)
        print(b)

    def test_MultiplyReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
  
        b = np.multiply.reduce(a1)
        print(b)

    def test_MultiplyReduce_INT32(self):

        a1 = np.arange(0, 100, dtype=np.int32).reshape((10,10));
  
        b = np.multiply.reduce(a1)
        print(b)

    def test_DivideReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        b = np.divide.reduce(a1)
        print(b)

    def test_RemainderReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));  

        b = np.remainder.reduce(a1)
        print(b)

    def test_FModReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        b = np.fmod.reduce(a1)
        print(b)

    def test_SquareReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));

        try :
            b = np.square.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ReciprocalReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));  

        try :
            b = np.reciprocal.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_OnesLikeReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));  

        try :
            b = np.ones_like.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_SqrtReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));        

        try :
            b = np.sqrt.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_NegativeReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));  

        try :
            b = np.negative.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

        
    def test_AbsoluteReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));

        try :
            b = np.absolute.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 

    def test_InvertReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));  

        try :
            b = np.invert.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LeftShiftReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        try :
            b = np.left_shift.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_RightShiftReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));  

        try :
            b = np.right_shift.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

            
    def test_BitwiseAndReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        try :
            b = np.bitwise_and.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

                        
    def test_BitwiseOrReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        try :
            b = np.bitwise_or.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_BitwiseXorReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
  
        try :
            b = np.bitwise_xor.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LessReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        b = np.less.reduce(a1)
        print(b)

    def test_LessEqualReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        b = np.less_equal.reduce(a1)
        print(b)

    def test_EqualReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));

        b = np.equal.reduce(a1)
        print(b)

    def test_NotEqualReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        b = np.not_equal.reduce(a1)
        print(b)

    def test_GreaterReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        b = np.greater.reduce(a1)
        print(b)

    def test_GreaterEqualReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        b = np.greater_equal.reduce(a1)
        print(b)

    def test_FloorDivideReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        b = np.floor_divide.reduce(a1)
        print(b)

    def test_TrueDivideReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        try:
            b = np.true_divide.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LogicalAndReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        b = np.logical_and.reduce(a1)
        print(b)

    def test_LogicalOrReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        b = np.logical_or.reduce(a1)
        print(b)

    def test_FloorReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));

        try :
            b = np.floor.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_CeilReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));

        try :
            b = np.ceil.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_MaximumReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        b = np.maximum.reduce(a1)
        print(b)

    def test_MinimumReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
 
        b = np.minimum.reduce(a1)
        print(b)
 
    def test_RintReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));

        try :
            b = np.rint.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ConjugateReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));

        try :
            b = np.conjugate.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_IsNANReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));

        try :
            b = np.isnan.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_FMaxReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        b = np.fmax.reduce(a1)
        print(b)

    def test_FMinReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
       
        b = np.fmin.reduce(a1)
        print(b)

    def test_HeavisideReduce_INT64(self):

        a1 = np.arange(0, 100, dtype=np.int64).reshape((10,10));
        
        try:
            b = np.heaviside.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
      #endregion
  
     
    #region ACCUMULATE Tests
    def test_AddAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
  
        b = np.add.accumulate(a1)
        print(b)

    def test_SubtractAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
        
        b = np.subtract.accumulate(a1)
        print(b)

    def test_MultiplyAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
  
        b = np.multiply.accumulate(a1)
        print(b)

    def test_DivideAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.divide.accumulate(a1)
        print(b)

    def test_RemainderAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        b = np.remainder.accumulate(a1)
        print(b)

    def test_FModAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.fmod.accumulate(a1)
        print(b)

    def test_SquareAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.square.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ReciprocalAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.reciprocal.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_OnesLikeAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.ones_like.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_SqrtAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.sqrt.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_NegativeAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.negative.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

        
    def test_AbsoluteAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.absolute.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 

    def test_InvertAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.invert.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LeftShiftAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        try :
            b = np.left_shift.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_RightShiftAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.right_shift.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

            
    def test_BitwiseAndAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        try :
            b = np.bitwise_and.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

                        
    def test_BitwiseOrAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        try :
            b = np.bitwise_or.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_BitwiseXorAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
  
        try :
            b = np.bitwise_xor.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LessAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.less.accumulate(a1)
        print(b)

    def test_LessEqualAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.less_equal.accumulate(a1)
        print(b)

    def test_EqualAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        b = np.equal.accumulate(a1)
        print(b)

    def test_NotEqualAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.not_equal.accumulate(a1)
        print(b)

    def test_GreaterAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.greater.accumulate(a1)
        print(b)

    def test_GreaterEqualAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.greater_equal.accumulate(a1)
        print(b)

    def test_FloorDivideAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.floor_divide.accumulate(a1)
        print(b)

    def test_TrueDivideAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.true_divide.accumulate(a1)
        print(b)

    def test_LogicalAndAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.logical_and.accumulate(a1)
        print(b)

    def test_LogicalOrAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.logical_or.accumulate(a1)
        print(b)

    def test_FloorAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.floor.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_CeilAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.ceil.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_MaximumAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.maximum.accumulate(a1)
        print(b)

    def test_MinimumAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
 
        b = np.minimum.accumulate(a1)
        print(b)
 
    def test_RintAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.rint.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ConjugateAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.conjugate.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_IsNANAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.isnan.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_FMaxAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.fmax.accumulate(a1)
        print(b)

    def test_FMinAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.fmin.accumulate(a1)
        print(b)

    def test_HeavisideAccumulate_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
        
        b = np.heaviside.accumulate(a1)
        print(b)
        
      #endregion
     
     
    #region REDUCEAT INT64 Tests
    def test_AddReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
  
        b = np.add.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_SubtractReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
        
        b = np.subtract.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_MultiplyReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
  
        b = np.multiply.reduceat(a1, [0, 2], axis = 1)
        print(b)
        
    def test_MultiplyReduceAt_INT32(self):

        a1 = np.arange(0, 9, dtype=np.int32).reshape((3,3));
  
        b = np.multiply.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_DivideReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.divide.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_RemainderReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        b = np.remainder.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_FModReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.fmod.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_SquareReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.square.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ReciprocalReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.reciprocal.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_OnesLikeReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.ones_like.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_SqrtReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.sqrt.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_NegativeReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.negative.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

        
    def test_AbsoluteReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.absolute.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 

    def test_InvertReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.invert.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LeftShiftReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        try :
            b = np.left_shift.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_RightShiftReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.right_shift.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

            
    def test_BitwiseAndReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        try :
            b = np.bitwise_and.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

                        
    def test_BitwiseOrReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        try :
            b = np.bitwise_or.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_BitwiseXorReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
  
        try :
            b = np.bitwise_xor.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LessReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.less.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_LessEqualReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.less_equal.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_EqualReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        b = np.equal.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_NotEqualReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.not_equal.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_GreaterReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.greater.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_GreaterEqualReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.greater_equal.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_FloorDivideReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.floor_divide.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_TrueDivideReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.true_divide.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_LogicalAndReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.logical_and.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_LogicalOrReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.logical_or.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_FloorReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.floor.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_CeilReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.ceil.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_MaximumReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.maximum.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_MinimumReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
 
        b = np.minimum.reduceat(a1, [0, 2], axis = 1)
        print(b)
 
    def test_RintReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.rint.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ConjugateReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.conjugate.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_IsNANReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));

        try :
            b = np.isnan.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_FMaxReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.fmax.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_FMinReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
       
        b = np.fmin.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_HeavisideReduceAt_INT64(self):

        a1 = np.arange(0, 9, dtype=np.int64).reshape((3,3));
        
        b = np.heaviside.reduceat(a1, [0, 2], axis = 1)
        print(b)
        
      #endregion
     

      #endregion 


if __name__ == '__main__':
    unittest.main()
