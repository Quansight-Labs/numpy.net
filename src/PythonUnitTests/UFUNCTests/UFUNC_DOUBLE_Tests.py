import unittest
import numpy as np

class Test_UFUNC_DOUBLE_Tests(unittest.TestCase):

    #region UFUNC DOUBLE Tests

    #region OUTER Tests
    def test_AddOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        b = np.add.outer(a1,a2)
        print(b)

    def test_SubtractOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        b = np.subtract.outer(a1,a2)
        print(b)

    def test_MultiplyOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        b = np.multiply.outer(a1,a2)
        print(b)

    def test_DivideOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        b = np.divide.outer(a1,a2)
        print(b)

    def test_RemainderOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        b = np.remainder.outer(a1,a2)
        print(b)

    def test_FModOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        b = np.fmod.outer(a1,a2)
        print(b)

    def test_SquareOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);

        try :
            b = np.square.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ReciprocalOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        try :
            b = np.reciprocal.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_OnesLikeOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        try :
            b = np.ones_like.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_SqrtOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
        
        try :
            b = np.sqrt.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_NegativeOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        try :
            b = np.negative.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

        
    def test_AbsoluteOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);

        try :
            b = np.absolute.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 

    def test_InvertOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        try :
            b = np.invert.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LeftShiftOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        try :
            b = np.left_shift.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_RightShiftOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        try :
            b = np.right_shift.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

            
    def test_BitwiseAndOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        try :
            b = np.bitwise_and.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

                        
    def test_BitwiseOrOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        try :
            b = np.bitwise_or.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_BitwiseXorOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        try :
            b = np.bitwise_xor.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LessOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        b = np.less.outer(a1,a2)
        print(b)

    def test_LessEqualOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        b = np.less_equal.outer(a1,a2)
        print(b)

    def test_EqualOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        b = np.equal.outer(a1,a2)
        print(b)

    def test_NotEqualOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        b = np.not_equal.outer(a1,a2)
        print(b)

    def test_GreaterOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        b = np.greater.outer(a1,a2)
        print(b)

    def test_GreaterEqualOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        b = np.greater_equal.outer(a1,a2)
        print(b)

    def test_FloorDivideOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        b = np.floor_divide.outer(a1,a2)
        print(b)

    def test_TrueDivideOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        b = np.true_divide.outer(a1,a2)
        print(b)

    def test_LogicalAndOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        b = np.logical_and.outer(a1,a2)
        print(b)

    def test_LogicalOrOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
  
        b = np.logical_or.outer(a1,a2)
        print(b)

    def test_FloorOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);

        try :
            b = np.floor.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_CeilOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);

        try :
            b = np.ceil.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_MaximumOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
 
        b = np.maximum.outer(a1,a2)
        print(b)

    def test_MinimumOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
 
        b = np.minimum.outer(a1,a2)
        print(b)
 
    def test_RintOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);

        try :
            b = np.rint.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ConjugateOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);

        try :
            b = np.conjugate.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_IsNANOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);

        try :
            b = np.isnan.outer(a1,a2)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_FMaxOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
 
        b = np.fmax.outer(a1,a2)
        print(b)

    def test_FMinOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
 
        b = np.fmin.outer(a1,a2)
        print(b)

    def test_HeavisideOuter_DOUBLE(self):

        a1 = np.arange(0, 5, dtype=np.float64);
        a2 = np.arange(3, 8, dtype=np.float64);
 
        b = np.heaviside.outer(a1,a2)
        print(b)
        
      #endregion
  

    #region REDUCE Tests
    def test_AddReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
  
        b = np.add.reduce(a1)
        print(b)

    def test_SubtractReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10)); 
        
        b = np.subtract.reduce(a1)
        print(b)

    def test_MultiplyReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
  
        b = np.multiply.reduce(a1)
        print(b)

    def test_DivideReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        b = np.divide.reduce(a1)
        print(b)

    def test_RemainderReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));  

        b = np.remainder.reduce(a1)
        print(b)

    def test_FModReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        b = np.fmod.reduce(a1)
        print(b)

    def test_SquareReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));

        try :
            b = np.square.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ReciprocalReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));  

        try :
            b = np.reciprocal.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_OnesLikeReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));  

        try :
            b = np.ones_like.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_SqrtReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));        

        try :
            b = np.sqrt.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_NegativeReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));  

        try :
            b = np.negative.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

        
    def test_AbsoluteReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));

        try :
            b = np.absolute.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 

    def test_InvertReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));  

        try :
            b = np.invert.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LeftShiftReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        try :
            b = np.left_shift.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_RightShiftReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));  

        try :
            b = np.right_shift.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

            
    def test_BitwiseAndReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        try :
            b = np.bitwise_and.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

                        
    def test_BitwiseOrReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        try :
            b = np.bitwise_or.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_BitwiseXorReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
  
        try :
            b = np.bitwise_xor.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LessReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        b = np.less.reduce(a1)
        print(b)

    def test_LessEqualReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        b = np.less_equal.reduce(a1)
        print(b)

    def test_EqualReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));

        b = np.equal.reduce(a1)
        print(b)

    def test_NotEqualReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        b = np.not_equal.reduce(a1)
        print(b)

    def test_GreaterReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        b = np.greater.reduce(a1)
        print(b)

    def test_GreaterEqualReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        b = np.greater_equal.reduce(a1)
        print(b)

    def test_FloorDivideReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        b = np.floor_divide.reduce(a1)
        print(b)

    def test_TrueDivideReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        b = np.true_divide.reduce(a1)
        print(b)

    def test_LogicalAndReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        b = np.logical_and.reduce(a1)
        print(b)

    def test_LogicalOrReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        b = np.logical_or.reduce(a1)
        print(b)

    def test_FloorReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));

        try :
            b = np.floor.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_CeilReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));

        try :
            b = np.ceil.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_MaximumReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        b = np.maximum.reduce(a1)
        print(b)

    def test_MinimumReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
 
        b = np.minimum.reduce(a1)
        print(b)
 
    def test_RintReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));

        try :
            b = np.rint.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ConjugateReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));

        try :
            b = np.conjugate.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_IsNANReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));

        try :
            b = np.isnan.reduce(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_FMaxReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        b = np.fmax.reduce(a1)
        print(b)

    def test_FMinReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
       
        b = np.fmin.reduce(a1)
        print(b)

    def test_HeavisideReduce_DOUBLE(self):

        a1 = np.arange(0, 100, dtype=np.float64).reshape((10,10));
        
        b = np.heaviside.reduce(a1)
        print(b)
        
      #endregion
  
     
    #region ACCUMULATE Tests
    def test_AddAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
  
        b = np.add.accumulate(a1)
        print(b)

    def test_SubtractAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
        
        b = np.subtract.accumulate(a1)
        print(b)

    def test_MultiplyAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
  
        b = np.multiply.accumulate(a1)
        print(b)

    def test_DivideAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.divide.accumulate(a1)
        print(b)

    def test_RemainderAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        b = np.remainder.accumulate(a1)
        print(b)

    def test_FModAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.fmod.accumulate(a1)
        print(b)

    def test_SquareAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.square.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ReciprocalAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.reciprocal.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_OnesLikeAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.ones_like.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_SqrtAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.sqrt.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_NegativeAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.negative.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

        
    def test_AbsoluteAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.absolute.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 

    def test_InvertAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.invert.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LeftShiftAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        try :
            b = np.left_shift.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_RightShiftAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.right_shift.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

            
    def test_BitwiseAndAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        try :
            b = np.bitwise_and.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

                        
    def test_BitwiseOrAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        try :
            b = np.bitwise_or.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_BitwiseXorAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
  
        try :
            b = np.bitwise_xor.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LessAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.less.accumulate(a1)
        print(b)

    def test_LessEqualAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.less_equal.accumulate(a1)
        print(b)

    def test_EqualAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        b = np.equal.accumulate(a1)
        print(b)

    def test_NotEqualAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.not_equal.accumulate(a1)
        print(b)

    def test_GreaterAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.greater.accumulate(a1)
        print(b)

    def test_GreaterEqualAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.greater_equal.accumulate(a1)
        print(b)

    def test_FloorDivideAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.floor_divide.accumulate(a1)
        print(b)

    def test_TrueDivideAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.true_divide.accumulate(a1)
        print(b)

    def test_LogicalAndAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.logical_and.accumulate(a1)
        print(b)

    def test_LogicalOrAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.logical_or.accumulate(a1)
        print(b)

    def test_FloorAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.floor.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_CeilAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.ceil.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_MaximumAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.maximum.accumulate(a1)
        print(b)

    def test_MinimumAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
 
        b = np.minimum.accumulate(a1)
        print(b)
 
    def test_RintAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.rint.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ConjugateAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.conjugate.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_IsNANAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.isnan.accumulate(a1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_FMaxAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.fmax.accumulate(a1)
        print(b)

    def test_FMinAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.fmin.accumulate(a1)
        print(b)

    def test_HeavisideAccumulate_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
        
        b = np.heaviside.accumulate(a1)
        print(b)
        
      #endregion
     
     
    #region REDUCEAT DOUBLE Tests
    def test_AddReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
  
        b = np.add.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_SubtractReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
        
        b = np.subtract.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_MultiplyReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
  
        b = np.multiply.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_DivideReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.divide.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_RemainderReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        b = np.remainder.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_FModReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.fmod.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_SquareReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.square.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ReciprocalReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.reciprocal.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_OnesLikeReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.ones_like.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_SqrtReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.sqrt.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_NegativeReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.negative.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

        
    def test_AbsoluteReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.absolute.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 

    def test_InvertReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.invert.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LeftShiftReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        try :
            b = np.left_shift.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")


    def test_RightShiftReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.right_shift.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

            
    def test_BitwiseAndReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        try :
            b = np.bitwise_and.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

                        
    def test_BitwiseOrReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        try :
            b = np.bitwise_or.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_BitwiseXorReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
  
        try :
            b = np.bitwise_xor.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_LessReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.less.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_LessEqualReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.less_equal.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_EqualReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        b = np.equal.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_NotEqualReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.not_equal.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_GreaterReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.greater.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_GreaterEqualReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.greater_equal.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_FloorDivideReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.floor_divide.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_TrueDivideReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.true_divide.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_LogicalAndReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.logical_and.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_LogicalOrReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.logical_or.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_FloorReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.floor.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_CeilReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.ceil.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_MaximumReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.maximum.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_MinimumReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
 
        b = np.minimum.reduceat(a1, [0, 2], axis = 1)
        print(b)
 
    def test_RintReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.rint.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_ConjugateReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.conjugate.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")
 
    def test_IsNANReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));

        try :
            b = np.isnan.reduceat(a1, [0, 2], axis = 1)
            print(b)
            self.fail("should have thrown exception")
        except:
            print("Exception occured")

    def test_FMaxReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.fmax.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_FMinReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
       
        b = np.fmin.reduceat(a1, [0, 2], axis = 1)
        print(b)

    def test_HeavisideReduceAt_DOUBLE(self):

        a1 = np.arange(0, 9, dtype=np.float64).reshape((3,3));
        
        b = np.heaviside.reduceat(a1, [0, 2], axis = 1)
        print(b)
        
      #endregion
     

      #endregion 


if __name__ == '__main__':
    unittest.main()
