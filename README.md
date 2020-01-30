
# NumpyDotNet

This is a 100% pure .NET implementation of the Numpy API.  This library is ported from the real Numpy source code. The C and the Python code of Numpy have been ported to C#.  This approach allows us to capture all of the nuances that are in the original Numpy libraries.

We have near 100% of the API ported and unit tested.  The one notable missing function is np.einsum (volunteers welcome).

The result is a .NET library that is 100% compatible with the Numpy API.  It can be run anywhere that .NET can run. There are no dependencies on having Python installed.  There are no performance issues related to interoping with Python. 

Since all of the underlying data are pure .NET System.Array types, they can be used like any other .NET array.

Our ndarray class is iterable, which means that it can be data bound to windows UI elements.

## Nuget packaging

The built release mode libraries are available from here:  

https://www.nuget.org/packages/NumpyDotNet/  

The unit tests that demonstrate how to use use all of the APIs and features are available here:  

https://www.nuget.org/packages/NumpyDotNet.UnitTests/  

The simple sample apps that shows windows console and GUI apps using NumpyDotNet:  

https://www.nuget.org/packages/NumpyDotNet.SampleApps/  



## Pure .NET data types
The underlying technology uses 100% .NET data types.   If you are working with doubles, then an array of doubles are allocated.  There is no worry about mismatching Python allocated C pointers to the .NET data type.  There is no worry about interop 'marshalling' of data and the complexities and problems that can cause.

## High performance calculations
We recently reworked our calculation engine to take advantage of the .NET Parallel libraries.  This has caused a massive improvement in performance/reduced calculation times. We are approaching the "C" based NumPy in performance now.  We hope to continue to fine tune the performance in future releases.

## Full multi-threading support
Unlike most NumPy implementations, our library does not require the GIL (Global Interpreter Lock).  This allows us to offer a fully multi-threaded library.  You are free to launch as many threads as you want to manipulate our ndarray objects. If your application can take advantage of that, you may be able to achieve much higher overall performance.

Take note that if multiple threads are manipulating the same ndarray object, you may get unexpected results.  For example, if one thread is adding numbers and another thread is dividing, you may get unexpected calculations.  The original authors of NumPy did a fine job of isolating arrays from each other to make this feature possible.  If you must manipulate the same ndarray from two different threads, it may be necessary to implement some sort of application layer locking on the ndarray object to get the expected results.


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
* System.Double
* System.Decimal (exclusive feature!)
* System.Numerics.Complex (exclusive feature!)
* System.Numerics.BigInteger (exclusive feature!)
* System.Object - A really cool feature! (exclusive feature!)

##### Future plans include support for:

* additional performance tunings
* System.String data type
* DateTime/TimeDiff data types ??
* User defined data types (System.Objects may make this unnecessary!)

## System.Objects - A really cool feature.

As the base class of all .NET classes, System.Object can hold anything.  

You can mix and match data types within the same ndarray such as Int32, Floats, BigInts, strings or custom data objects.  If you try to perform math or comparision operations on these objects we do our best effort to make the operation work.  If you try something like dividing a string by a integer, the system will throw an exception to let you know it can't be done.  

Most NumPy operations have been shown to work really well with these object arrays.  We were pleasantly suprised by how well it worked. Check out the latest unit tests for examples of what can be done.

It is very possible to build custom data types and have them processed by the NumpyDotNet system. If you want the ability to add an integer to your custom data object or to add two custom data objects together, simple define your custom data object class to overide the + sign. This holds true for all of the operators allowed by the C# language (https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/operators/operator-overloading). 

Please see our latest unit tests and sample apps for examples on how to build custom data types with object arrays.




## Accessing the underlying array

We have extended the Numpy API to allow you to access the underlying System.Array type data.

    ndarray O = np.arange(0, 12, dtype: np.Int32);  // create Int32 ndarray   
    ndarray A = (ndarray)O["2:6"];                  // create a view of this middle 4 elements, starting at index 2   

    Int32[] O1 = O.AsInt32Array();                  // Int32[] {0,1,2,3,4,5,6,7,8,9,10,11}  reference to real data   
    Int32[] A1 = A.AsInt32Array();                  // Int32[] {2,3,4,5}  The view was copied into a new array.   
    Int16[] A2 = A.AsInt16Array();                  // Int16[] {2,3,4,5}  The view was copied into a new array.   

 
## Accessing a scalar return value

Many Numpy operations can return an ndarray full of data, or a single scalar object if the resultant is a single item.  Python, not being a strongly typed language, can get away with that.  .NET languages can't. .NET functions need to specify the exact return type.  In most cases, our API will always return an ndarray, even if the resultant is a single data item. To help with this issue, we have extended the ndarray class to have the following APIs.  


    ndarray A = np.array(new int[] {1,2,3});  
    
    int I1 = (int)A.GetItem(1);  // I1 = 2;  
    A.SetItem(99, 1);  
    int I2 = (int)A.GetItem(1);  // I2 = 99;  


## Array Slicing

Numpy allows you to create different views of an array using a technique called ([array slicing](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#arrays-indexing)).  As an interpreted language, python can use syntax that C# can't.  This necessitates a small difference in NumpyDotNet.

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


