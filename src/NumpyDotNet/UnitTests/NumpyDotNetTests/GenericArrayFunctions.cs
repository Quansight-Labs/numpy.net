using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumpyLib;

/// <summary>
/// These tests are used to experiment with adding a ndarray<T> generics wrapper around the existing ndarray class.
/// There may be some cases where such a wrapper could be helpful to application writers.
/// This is a work in progress.
/// </summary>
namespace NumpyDotNetTests
{
    public class ndarray<T>
    {
        public T[] data
        {
            get
            {
                return arr.ToArray<T>();
            }
        }

        private ndarray arr;

        public ndarray()
        {

        }
        public ndarray(ndarray arr)
        {
            this.arr = arr;
        }

        public static ndarray<T> array(T[] data)
        {
            ndarray arr = np.array(data);
            ndarray<T> tarr = new ndarray<T>(arr);
            return tarr;
        }
        public static ndarray<T> array(T[,] data)
        {
            ndarray arr = np.array(data);
            ndarray<T> tarr = new ndarray<T>(arr);
            return tarr;
        }

        public ndarray<T> Sum(int? axis = null)
        {
            var x = arr.Sum(axis);

            ndarray<T> ret = new ndarray<T>(x);
            return ret;
        }
    }

    [TestClass]
    public class GenericArrayFunctions
    {
        [TestMethod]
        public void test_generic_array_1()
        {
            var ta = ndarray<Int32>.array(new Int32[] { 1, 2, 3 });

            var data = ta.data;
            var sum = ta.Sum();
        }

        [TestMethod]
        public void test_generic_array_2()
        {
            var ta = ndarray<Int32>.array(new Int32[,] { { 1, 2, 3 }, { 1, 2, 3 } });

            var data = ta.data;
            var sum0 = ta.Sum(0);
            var sum1 = ta.Sum(1);
        }
    }
}
