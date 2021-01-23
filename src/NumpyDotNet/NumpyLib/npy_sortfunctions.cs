/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2018-2019
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyLib
{
    internal class Sort_BOOL : SortBase<bool>
    {
        public override int CompareTo(bool d1, bool d2)
        {
            if (d1 == d2)
                return 0;
            if (d2)
                return -1;
            return 1;
        }
    }
    internal class Sort_BYTE : SortBase<sbyte>
    {
        public override int CompareTo(sbyte d1, sbyte d2)
        {
            if (d1 == d2)
                return 0;
            if (d1 < d2)
                return -1;
            return 1;
        }
    }
    internal class Sort_UBYTE : SortBase<byte>
    {
        public override int CompareTo(byte d1, byte d2)
        {
            if (d1 == d2)
                return 0;
            if (d1 < d2)
                return -1;
            return 1;
        }
    }
    internal class Sort_INT16 : SortBase<short>
    {
        public override int CompareTo(Int16 d1, Int16 d2)
        {
            if (d1 == d2)
                return 0;
            if (d1 < d2)
                return -1;
            return 1;
        }
    }
    internal class Sort_UINT16 : SortBase<UInt16>
    {
        public override int CompareTo(UInt16 d1, UInt16 d2)
        {
            if (d1 == d2)
                return 0;
            if (d1 < d2)
                return -1;
            return 1;
        }
    }
    internal class Sort_INT32 : SortBase<Int32>
    {
        public override int CompareTo(Int32 d1, Int32 d2)
        {
            if (d1 == d2)
                return 0;
            if (d1 < d2)
                return -1;
            return 1;
        }
    }
    internal class Sort_UINT32 : SortBase<UInt32>
    {
        public override int CompareTo(UInt32 d1, UInt32 d2)
        {
            if (d1 == d2)
                return 0;
            if (d1 < d2)
                return -1;
            return 1;
        }
    }
    internal class Sort_INT64 : SortBase<Int64>
    {
        public override int CompareTo(Int64 d1, Int64 d2)
        {
            if (d1 == d2)
                return 0;
            if (d1 < d2)
                return -1;
            return 1;
        }
    }
    internal class Sort_UINT64 : SortBase<UInt64>
    {
        public override int CompareTo(UInt64 d1, UInt64 d2)
        {
            if (d1 == d2)
                return 0;
            if (d1 < d2)
                return -1;
            return 1;
        }
    }
    internal class Sort_FLOAT : SortBase<float>
    {
        public override int CompareTo(float d1, float d2)
        {
            if (d1 == d2)
                return 0;
            if (d1 < d2)
                return -1;
            return 1;
        }
    }
    internal class Sort_DOUBLE : SortBase<double>
    {
        public override int CompareTo(double d1, double d2)
        {
            if (d1 == d2)
                return 0;
            if (d1 < d2)
                return -1;
            return 1;
        }
    }
    internal class Sort_DECIMAL : SortBase<decimal>
    {
        public override int CompareTo(decimal d1, decimal d2)
        {
            if (d1 == d2)
                return 0;
            if (d1 < d2)
                return -1;
            return 1;
        }
    }
    internal class Sort_COMPLEX
    {
        class ArgSortData_COMPLEX : IComparable
        {
            System.Numerics.Complex dvalue;
            public npy_intp index;

            public ArgSortData_COMPLEX(npy_intp index, System.Numerics.Complex d)
            {
                this.index = index;
                this.dvalue = d;
            }

            public int CompareTo(object obj)
            {
                ArgSortData_COMPLEX cv = obj as ArgSortData_COMPLEX;
                return this.dvalue.Real.CompareTo(cv.dvalue.Real);
            }
        }

        public void argSortIndexes(VoidPtr ip, npy_intp m, VoidPtr sortData, npy_intp startingIndex, NPY_SORTKIND kind, int DivSize, int IntpDivSize)
        {
            var data = sortData.datap as System.Numerics.Complex[];

            var argSortDouble = new ArgSortData_COMPLEX[m];

            var adjustedIndex = startingIndex + (sortData.data_offset >> DivSize);

            for (int i = 0; i < m; i++)
            {
                argSortDouble[i] = new ArgSortData_COMPLEX(i, data[i + adjustedIndex]);
            }

            //Array.Sort(argSortDouble);
            argSortDouble = argSortDouble.AsParallel().OrderBy(t => t).ToArray();

            npy_intp[] _ip = (npy_intp[])ip.datap;

            for (int i = 0; i < m; i++)
            {
                _ip[i + (ip.data_offset >> IntpDivSize)] = argSortDouble[i].index - startingIndex;
            }
        }
    }
    internal class Sort_BIGINT : SortBase<System.Numerics.BigInteger>
    {
        public override int CompareTo(System.Numerics.BigInteger d1, System.Numerics.BigInteger d2)
        {
            if (d1 == d2)
                return 0;
            if (d1 < d2)
                return -1;
            return 1;
        }
    }
    internal class Sort_OBJECT
    {
        class ArgSortData_OBJECT : IComparable
        {
            dynamic dvalue;
            public npy_intp index;

            public ArgSortData_OBJECT(npy_intp index, System.Object d)
            {
                this.index = index;
                this.dvalue = d;
            }

            public int CompareTo(object obj)
            {
                ArgSortData_OBJECT cv = obj as ArgSortData_OBJECT;
                return this.dvalue.CompareTo(cv.dvalue);
            }
        }

        public void argSortIndexes(VoidPtr ip, npy_intp m, VoidPtr sortData, npy_intp startingIndex, NPY_SORTKIND kind, int DivSize, int IntpDivSize)
        {
            var data = sortData.datap as System.Object[];

            var argSortDouble = new ArgSortData_OBJECT[m];

            var adjustedIndex = startingIndex + (sortData.data_offset >> DivSize);

            for (int i = 0; i < m; i++)
            {
                argSortDouble[i] = new ArgSortData_OBJECT(i, data[i + adjustedIndex]);
            }

            //Array.Sort(argSortDouble);
            argSortDouble = argSortDouble.AsParallel().OrderBy(t => t).ToArray();

            npy_intp[] _ip = (npy_intp[])ip.datap;

            for (int i = 0; i < m; i++)
            {
                _ip[i + (ip.data_offset >> IntpDivSize)] = argSortDouble[i].index - startingIndex;
            }
        }
    }
    internal class Sort_STRING : SortBase<System.String>
    {
        public override int CompareTo(System.String d1, System.String d2)
        {
            return d1.CompareTo(d2);
        }
    }
    internal abstract class SortBase<T> where T : IComparable
    {
        // disable threading.  Seems to be slower
        const bool UseMergeSortThreading = false;

        // use struct for merge sorts.  It is faster.
        protected struct ArgSortData
        {
            public T dvalue;
            public npy_intp index;
        }

        // use class for quick sort.  It is faster.
        class CArgSortData : IComparable
        {
            T dvalue;
            public npy_intp index;

            public CArgSortData(npy_intp index, T d)
            {
                this.index = index;
                this.dvalue = d;
            }

            public int CompareTo(object obj)
            {
                CArgSortData cv = obj as CArgSortData;
                return this.dvalue.CompareTo(cv.dvalue);
            }
        }

        public abstract int CompareTo(T d1, T d2);

        public void argSortIndexes(VoidPtr ip, npy_intp m, VoidPtr sortData, npy_intp startingIndex, NPY_SORTKIND kind, int DivSize, int IntpDivSize)
        {
            if (kind == NPY_SORTKIND.NPY_MERGESORT)
            {
                argMergeSortIndexes(ip, m, sortData, startingIndex, DivSize, IntpDivSize);
                return;
            }

            if (kind == NPY_SORTKIND.NPY_QUICKSORT)
            {
                argQuickSortIndexes(ip, m, sortData, startingIndex, DivSize, IntpDivSize);
                return;
            }

            if (kind == NPY_SORTKIND.NPY_HEAPSORT)
            {
                throw new Exception("HEAPSORT not supported");
            }

            throw new Exception("Unrecognized sort type");

    
        }

        #region QuickSort
        private void argQuickSortIndexes(VoidPtr ip, npy_intp m, VoidPtr sortData, npy_intp startingIndex, int DivSize, int IntpDivSize)
        {
            T[] data = sortData.datap as T[];

            var argSortData = new CArgSortData[m];

            var adjustedIndex = startingIndex + (sortData.data_offset >> DivSize);

            for (npy_intp i = 0; i < m; i++)
            {
                argSortData[i] = new CArgSortData(i, data[adjustedIndex++]);
            }

            //Array.Sort(argSortData);
            argSortData = argSortData.AsParallel().OrderBy(t => t).ToArray();

            npy_intp[] _ip = (npy_intp[])ip.datap;

            npy_intp data_offset = ip.data_offset >> IntpDivSize;
            for (int i = 0; i < m; i++)
            {
                _ip[data_offset++] = argSortData[i].index - startingIndex;
            }
        }
        #endregion

        #region MergeSort
        private void argMergeSortIndexes(VoidPtr ip, npy_intp m, VoidPtr sortData, npy_intp startingIndex, int DivSize, int IntpDivSize)
        {
            T[] data = sortData.datap as T[];

            var argSortData = new ArgSortData[m];

            var adjustedIndex = startingIndex + (sortData.data_offset >> DivSize);

            for (npy_intp i = 0; i < m; i++)
            {
                argSortData[i].index = i;
                argSortData[i].dvalue = data[adjustedIndex++];
            }

            ArgMergeSort(argSortData, 0, m - 1);

            npy_intp[] _ip = (npy_intp[])ip.datap;

            npy_intp data_offset = ip.data_offset >> IntpDivSize;
            for (int i = 0; i < m; i++)
            {
                _ip[data_offset++] = argSortData[i].index - startingIndex;
            }
        }

        private void ArgMergeSort(ArgSortData[] input, npy_intp left, npy_intp right)
        {
            int depthRemaining = 0;
            if (UseMergeSortThreading)
            {
                depthRemaining = (int)Math.Log(Environment.ProcessorCount, 2) + 4;
            }
           

            _ArgMergeSort(input, left, right, depthRemaining);
        }

        private void _ArgMergeSort(ArgSortData[] input, npy_intp left, npy_intp right, int depthRemaining)
        {
            if (left < right)
            {
                npy_intp middle = (left + right) / 2;

                if (UseMergeSortThreading && depthRemaining > 0)
                {
                    var t1 = Task.Run(() =>
                    {
                        _ArgMergeSort(input, left, middle, depthRemaining - 1);
                    });
                    _ArgMergeSort(input, middle + 1, right, depthRemaining - 1);
                    Task.WaitAll(t1);
                }
                else
                {
                    _ArgMergeSort(input, left, middle, depthRemaining - 1);
                    _ArgMergeSort(input, middle + 1, right, depthRemaining - 1);
                }

                _ArgMerge(input, left, middle, right);
            }
        }

        protected virtual void _ArgMerge(ArgSortData[] input, npy_intp left, npy_intp middle, npy_intp right)
        {
            var leftArray = new ArgSortData[middle - left + 1];
            var rightArray = new ArgSortData[right - middle];

            Array.Copy(input, left, leftArray, 0, middle - left + 1);
            Array.Copy(input, middle + 1, rightArray, 0, right - middle);

            int i = 0;
            int j = 0;
            for (npy_intp k = left; k < right + 1; k++)
            {
                if (i == leftArray.Length)
                {
                    input[k] = rightArray[j];
                    j++;
                }
                else if (j == rightArray.Length)
                {
                    input[k] = leftArray[i];
                    i++;
                }
                else if (CompareTo(leftArray[i].dvalue, rightArray[j].dvalue) <= 0)
                {
                    input[k] = leftArray[i];
                    i++;
                }
                else
                {
                    input[k] = rightArray[j];
                    j++;
                }
            }
        }
        #endregion
    }

}
