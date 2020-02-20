import unittest
import numpy as np

class Test_UFUNC_UINT64(unittest.TestCase):

    #region UFUNC UINT64 Tests

    #region OUTER Tests
    def test_AddOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        b = np.add.outer(a1,a2)
        print(b)

    def test_SubtractOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        b = np.subtract.outer(a1,a2)
        print(b)

    def test_MultiplyOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        b = np.multiply.outer(a1,a2)
        print(b)

    def test_DivideOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        b = np.divide.outer(a1,a2)
        print(b)

    def test_RemainderOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        b = np.remainder.outer(a1,a2)
        print(b)

    def test_FModOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        b = np.fmod.outer(a1,a2)
        print(b)

    def test_SquareOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);

        try :
            b = np.square.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ReciprocalOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        try :
            b = np.reciprocal.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_OnesLikeOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        try :
            b = np.ones_like.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_SqrtOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
        
        try :
            b = np.sqrt.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_NegativeOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        try :
            b = np.negative.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

        
    def test_AbsoluteOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);

        try :
            b = np.absolute.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 

    def test_InvertOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        try :
            b = np.invert.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LeftShiftOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        try :
            b = np.left_shift.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_RightShiftOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        try :
            b = np.right_shift.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

            
    def test_BitwiseAndOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        try :
            b = np.bitwise_and.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

                        
    def test_BitwiseOrOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        try :
            b = np.bitwise_or.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_BitwiseXorOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        try :
            b = np.bitwise_xor.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LessOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        b = np.less.outer(a1,a2)
        print(b)

    def test_LessEqualOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        b = np.less_equal.outer(a1,a2)
        print(b)

    def test_EqualOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        b = np.equal.outer(a1,a2)
        print(b)

    def test_NotEqualOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        b = np.not_equal.outer(a1,a2)
        print(b)

    def test_GreaterOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        b = np.greater.outer(a1,a2)
        print(b)

    def test_GreaterEqualOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        b = np.greater_equal.outer(a1,a2)
        print(b)

    def test_FloorDivideOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        b = np.floor_divide.outer(a1,a2)
        print(b)

    def test_TrueDivideOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        b = np.true_divide.outer(a1,a2)
        print(b)

    def test_LogicalAndOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        b = np.logical_and.outer(a1,a2)
        print(b)

    def test_LogicalOrOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
  
        b = np.logical_or.outer(a1,a2)
        print(b)

    def test_FloorOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);

        try :
            b = np.floor.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_CeilOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);

        try :
            b = np.ceil.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_MaximumOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
 
        b = np.maximum.outer(a1,a2)
        print(b)

    def test_MinimumOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
 
        b = np.minimum.outer(a1,a2)
        print(b)
 
    def test_RintOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);

        try :
            b = np.rint.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ConjugateOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);

        try :
            b = np.conjugate.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_IsNANOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);

        try :
            b = np.isnan.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_FMaxOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
 
        b = np.fmax.outer(a1,a2)
        print(b)

    def test_FMinOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
 
        b = np.fmin.outer(a1,a2)
        print(b)

    def test_HeavisideOuter_UINT64(self):

        a1 = np.arange(0, 5, dtype=np.uint64);
        a2 = np.arange(3, 8, dtype=np.uint64);
 
        b = np.heaviside.outer(a1,a2)
        print(b)
        
      #endregion
  

    #region REDUCE Tests
    def test_AddReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
  
        b = np.add.reduce(a1)
        print(b)

    def test_SubtractReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10)); 
        
        b = np.subtract.reduce(a1)
        print(b)

    def test_MultiplyReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
  
        b = np.multiply.reduce(a1)
        print(b)

    def test_DivideReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        b = np.divide.reduce(a1)
        print(b)

    def test_RemainderReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));  

        b = np.remainder.reduce(a1)
        print(b)

    def test_FModReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        b = np.fmod.reduce(a1)
        print(b)

    def test_SquareReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));

        try :
            b = np.square.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ReciprocalReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));  

        try :
            b = np.reciprocal.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_OnesLikeReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));  

        try :
            b = np.ones_like.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_SqrtReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));        

        try :
            b = np.sqrt.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_NegativeReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));  

        try :
            b = np.negative.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

        
    def test_AbsoluteReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));

        try :
            b = np.absolute.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 

    def test_InvertReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));  

        try :
            b = np.invert.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LeftShiftReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        try :
            b = np.left_shift.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_RightShiftReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));  

        try :
            b = np.right_shift.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

            
    def test_BitwiseAndReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        try :
            b = np.bitwise_and.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

                        
    def test_BitwiseOrReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        try :
            b = np.bitwise_or.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_BitwiseXorReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
  
        try :
            b = np.bitwise_xor.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LessReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        b = np.less.reduce(a1)
        print(b)

    def test_LessEqualReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        b = np.less_equal.reduce(a1)
        print(b)

    def test_EqualReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));

        b = np.equal.reduce(a1)
        print(b)

    def test_NotEqualReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        b = np.not_equal.reduce(a1)
        print(b)

    def test_GreaterReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        b = np.greater.reduce(a1)
        print(b)

    def test_GreaterEqualReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        b = np.greater_equal.reduce(a1)
        print(b)

    def test_FloorDivideReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        b = np.floor_divide.reduce(a1)
        print(b)

    def test_TrueDivideReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        try:
            b = np.true_divide.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LogicalAndReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        b = np.logical_and.reduce(a1)
        print(b)

    def test_LogicalOrReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        b = np.logical_or.reduce(a1)
        print(b)

    def test_FloorReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));

        try :
            b = np.floor.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_CeilReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));

        try :
            b = np.ceil.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_MaximumReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        b = np.maximum.reduce(a1)
        print(b)

    def test_MinimumReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
 
        b = np.minimum.reduce(a1)
        print(b)
 
    def test_RintReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));

        try :
            b = np.rint.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ConjugateReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));

        try :
            b = np.conjugate.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_IsNANReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));

        try :
            b = np.isnan.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_FMaxReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        b = np.fmax.reduce(a1)
        print(b)

    def test_FMinReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
       
        b = np.fmin.reduce(a1)
        print(b)

    def test_HeavisideReduce_UINT64(self):

        a1 = np.arange(0, 100, dtype=np.uint64).reshape((10,10));
        
        try:
            b = np.heaviside.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

      #endregion
  
     
    #region ACCUMULATE Tests
    def test_AddAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
  
        b = np.add.accumulate(a1)
        print(b)

    def test_SubtractAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
        
        b = np.subtract.accumulate(a1)
        print(b)

    def test_MultiplyAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
  
        b = np.multiply.accumulate(a1)
        print(b)

    def test_DivideAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.divide.accumulate(a1)
        print(b)

    def test_RemainderAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        b = np.remainder.accumulate(a1)
        print(b)

    def test_FModAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.fmod.accumulate(a1)
        print(b)

    def test_SquareAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.square.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ReciprocalAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.reciprocal.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_OnesLikeAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.ones_like.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_SqrtAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.sqrt.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_NegativeAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.negative.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

        
    def test_AbsoluteAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.absolute.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 

    def test_InvertAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.invert.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LeftShiftAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        try :
            b = np.left_shift.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_RightShiftAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.right_shift.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

            
    def test_BitwiseAndAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        try :
            b = np.bitwise_and.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

                        
    def test_BitwiseOrAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        try :
            b = np.bitwise_or.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_BitwiseXorAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
  
        try :
            b = np.bitwise_xor.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LessAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.less.accumulate(a1)
        print(b)

    def test_LessEqualAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.less_equal.accumulate(a1)
        print(b)

    def test_EqualAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        b = np.equal.accumulate(a1)
        print(b)

    def test_NotEqualAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.not_equal.accumulate(a1)
        print(b)

    def test_GreaterAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.greater.accumulate(a1)
        print(b)

    def test_GreaterEqualAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.greater_equal.accumulate(a1)
        print(b)

    def test_FloorDivideAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.floor_divide.accumulate(a1)
        print(b)

    def test_TrueDivideAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.true_divide.accumulate(a1)
        print(b)

    def test_LogicalAndAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.logical_and.accumulate(a1)
        print(b)

    def test_LogicalOrAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.logical_or.accumulate(a1)
        print(b)

    def test_FloorAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.floor.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_CeilAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.ceil.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_MaximumAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.maximum.accumulate(a1)
        print(b)

    def test_MinimumAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
 
        b = np.minimum.accumulate(a1)
        print(b)
 
    def test_RintAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.rint.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ConjugateAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.conjugate.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_IsNANAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.isnan.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_FMaxAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.fmax.accumulate(a1)
        print(b)

    def test_FMinAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.fmin.accumulate(a1)
        print(b)

    def test_HeavisideAccumulate_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
        
        b = np.heaviside.accumulate(a1)
        print(b)
        
      #endregion
     
     
    #region REDUCEAT UINT64 Tests
    def test_AddReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
  
        b = np.add.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_SubtractReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
        
        b = np.subtract.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_MultiplyReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
  
        b = np.multiply.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_DivideReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.divide.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_RemainderReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        b = np.remainder.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_FModReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.fmod.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_SquareReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.square.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ReciprocalReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.reciprocal.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_OnesLikeReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.ones_like.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_SqrtReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.sqrt.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_NegativeReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.negative.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

        
    def test_AbsoluteReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.absolute.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 

    def test_InvertReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.invert.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LeftShiftReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        try :
            b = np.left_shift.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_RightShiftReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.right_shift.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

            
    def test_BitwiseAndReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        try :
            b = np.bitwise_and.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

                        
    def test_BitwiseOrReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        try :
            b = np.bitwise_or.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_BitwiseXorReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
  
        try :
            b = np.bitwise_xor.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LessReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.less.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_LessEqualReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.less_equal.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_EqualReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        b = np.equal.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_NotEqualReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.not_equal.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_GreaterReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.greater.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_GreaterEqualReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.greater_equal.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_FloorDivideReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.floor_divide.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_TrueDivideReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.true_divide.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_LogicalAndReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.logical_and.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_LogicalOrReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.logical_or.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_FloorReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.floor.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_CeilReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.ceil.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_MaximumReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.maximum.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_MinimumReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
 
        b = np.minimum.reduceat(a1, [0, 2], axis = 1)
        print(b)
 
    def test_RintReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.rint.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ConjugateReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.conjugate.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_IsNANReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));

        try :
            b = np.isnan.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_FMaxReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.fmax.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_FMinReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
       
        b = np.fmin.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_HeavisideReduceAt_UINT64(self):

        a1 = np.arange(0, 9, dtype=np.uint64).reshape((3,3));
        
        b = np.heaviside.reduceat(a1, [0, 2], axis = 1)
        print(b)
        
      #endregion
     

      #endregion 


if __name__ == '__main__':
    unittest.main()
