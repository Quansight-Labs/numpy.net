
# NumpyDotNet

This is a 100% pure .NET implementation of the Numpy API.  This library is ported from the real Numpy source code. The C and the Python code of Numpy have been ported to C#.  

We have near 100% of the API ported and unit tested.  The one notable missing is np.einsum (volunteers welcome).

The result is a .NET library that is 100% compatible with the Numpy API.  It can be run anywhere that .NET can run. There are no dependencies on having Python installed.  There are no performance issues related to interoping with Python. 

Since all of the underlying data are pure .NET System.Array types, they can be used like any other array.

Our ndarray class is iterable, which means that it can be data bound to windows UI elements.


## Pure .NET data types
The underlying technology uses 100% .NET data types.   If you are working with doubles, then an array of doubles are allocated.  There is no worry about mismatching Python allocated C pointers to the .NET data type.  There is no worry about interop 'marshalling' of data and the complexities and problems that can cause.

##### Our API has full support of the following .NET data types:

* System.Boolean
* System.Sbyte
* System.Byte
* System.UInt16
* System.Int16
* System.UInt32
* System.Int32
* System.UInt64
* System.Int64
* System.Single (float)
* System.Double.

##### Currently we have partial support of the following .NET data types:

* System.Decimal

##### Future plans include support for:

* System.Numerics.Complex
* System.Numerics.BigInteger


## Array Slicing:

Numpy allows you to create different views of an array using a technique called "slicing" ([array slicing](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#arrays-indexing)).  As an interpreted language, python can use syntax that the compilable C# can't.  This necessitates a difference in NumpyDotNet.

In the example of python slicing array like this:  

    A1 = A[1:4:2, 10:0:-2]    

NumpyDotNet supports the slicing syntax  like this:  

    var A1 = A["1:4:2", "10:0:-2"];  

or like this:  

    var A1 = A[new Slice(1,4,2), new Slice(10,2,-2)];  

We also support Ellipsis slicing:  

    var A1 = A["..."];  



## Documentation

We have worked hard to make NumpyDotNET as similar to the python NumPy as possible.  We rely on the official [NumPy manual](https://docs.scipy.org/doc/numpy/). 


