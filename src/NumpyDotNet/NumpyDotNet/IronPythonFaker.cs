/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2018-2021
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
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using NumpyDotNet;
#if NPY_INTP_64
using npy_intp = System.Int64;
using npy_ucs4 = System.Int64;
#else
using npy_intp = System.Int32;
using npy_ucs4 = System.Int32;
#endif

namespace NumpyDotNet
{
    /// <summary>
    /// public class to get optimized
    /// </summary>
    internal sealed class TupleEnumerator : IEnumerable, IEnumerator, IEnumerator<object>
    {
        private int _curIndex;
        private PythonTuple _tuple;

        public TupleEnumerator(PythonTuple t)
        {
            _tuple = t;
            _curIndex = -1;
        }

        #region IEnumerator Members

        public object Current
        {
            get
            {
                // access _data directly because this is what CPython does:
                // class T(tuple):
                //     def __getitem__(self): return None
                // 
                // for x in T((1,2)): print x
                // prints 1 and 2
                return _tuple._data[_curIndex];
            }
        }

        public bool MoveNext()
        {
            if ((_curIndex + 1) >= _tuple.Count)
            {
                return false;
            }
            _curIndex++;
            return true;
        }

        public void Reset()
        {
            _curIndex = -1;
        }

        #endregion

        #region IDisposable Members

        public void Dispose()
        {
            GC.SuppressFinalize(this);
        }

        #endregion

        #region IEnumerable Members

        public IEnumerator GetEnumerator()
        {
            return this;
        }

        #endregion
    }

    internal class PythonTuple  : IReadOnlyList<object>
    {
        internal readonly object[] _data;

        public PythonTuple()
        {
            this._data = new object[0];
        }

        public PythonTuple(object o)
        {
            this._data = MakeItems(o);
        }

        public PythonTuple(object[] items)
        {
            this._data = items;
        }
 
        public object this[BigInteger index]
        {
            get
            {
                return this[(int)index];
            }
        }

        //public object this[object index]
        //{
        // get {
        //        return this[Converter.ConvertToIndex(index)];
        //    }
        //}

        public object this[int index]
        {
            get
            {
                if (index < 0 || index >= _data.Length)
                {
                    throw new Exception("index out of range");
                }
                return _data[index];
            }
        }

        //public virtual object this[Slice slice]
        //{
        //    get
        //    {
        //        int start, stop, step;
        //        slice.indices(_data.Length, out start, out stop, out step);

        //        if (start == 0 && stop == _data.Length && step == 1 &&
        //            this.GetType() == typeof(PythonTuple))
        //        {
        //            return this;
        //        }
        //        return MakeTuple(ArrayOps.GetSlice(_data, start, stop, step));
        //    }
        //}



        public int Count
        {
            get { return _data.Length; }
        }

        public virtual IEnumerator __iter__()
        {
            return new TupleEnumerator(this);
        }

        #region IEnumerable Members

        public IEnumerator GetEnumerator()
        {
            return __iter__();
        }

        IEnumerator<object> IEnumerable<object>.GetEnumerator()
        {
            return new TupleEnumerator(this);
        }

        #endregion

        public override string ToString()
        {
            StringBuilder buf = new StringBuilder();
            buf.Append("(");
            for (int i = 0; i < _data.Length; i++)
            {
                if (i > 0) buf.Append(", ");
                buf.Append(_data[i].ToString());
            }
            if (_data.Length == 1) buf.Append(",");
            buf.Append(")");
            return buf.ToString();
        }


        private static object[] MakeItems(object o)
        {
            object[] arr;
            if (o is PythonTuple)
            {
                return ((PythonTuple)o)._data;
            }
            else if (o is string)
            {
                string s = (string)o;
                object[] res = new object[s.Length];
                var sarray = s.Select(x => new string(x, 1)).ToArray();
                for (int i = 0; i < res.Length; i++)
                {
                    res[i] = sarray[i];
                }
                return res;
            }
            else if (o is IList)
            {
                return new object[0];
                //return ((IList)o)
            }
            else if ((arr = o as object[]) != null)
            {
                var _data = new object[arr.Length];
                Array.Copy(arr, _data, arr.Length);
                return _data;
            }
            else
            {
                //List<object> l = new List<object>();
                //IEnumerator i = PythonOps.GetEnumerator(o);
                //while (i.MoveNext())
                //{
                //    l.Add(i.Current);
                //}

                //return l.ToArray();

                return new object[0];
            }
        }
    }

  
    public class Ellipsis
    {
        internal static Ellipsis Instance = new Ellipsis();
    }

    public interface ISlice
    {
        object Start
        {
            get;
            set;
        }
        object Stop
        {
            get;
            set;
        }
        object Step
        {
            get;
            set;
        }
    }

    public class Slice : ISlice
    {
        public object stop;
        public object start;
        public object step;

        public Slice(object a, object b = null, object c = null)
        {
            start = a;
            stop = b;
            step = c;
        }

        public object Start
        {
            get { return start; }
            set { start = value; }
        }
        public object Stop
        {
            get { return stop; }
            set { stop = value; }
        }
        public object Step
        {
            get { return step; }
            set { step = value; }
        }
    }

    internal class PythonOps
    {
        internal static int Length(object src)
        {
            return 0;
        }

    }

    internal class ArgumentTypeException : Exception
    {
        public ArgumentTypeException(string message) : base(message)
        {

        }
    }

    internal class TypeErrorException : Exception
    {
        public TypeErrorException(string message) : base(message)
        {

        }
    }


    internal class FloatingPointException : Exception
    {
        public FloatingPointException(string message) : base(message)
        {

        }
    }

    internal class RuntimeException : Exception
    {
        public RuntimeException(string message) : base(message)
        {

        }
    }

    internal class ValueError : Exception
    {
        public ValueError(string message) : base(message)
        {

        }
    }

    internal class ZeroDivisionError : Exception
    {
        public ZeroDivisionError(string message) : base(message)
        {

        }
    }


    internal class TypeError : Exception
    {
        public TypeError(string message) : base(message)
        {

        }
    }

    internal class AxisError : Exception
    {
        public AxisError(string message) : base(message)
        {

        }
    }

    internal class RuntimeError : Exception
    {
        public RuntimeError(string message) : base(message)
        {

        }
    }


    internal static class PythonFunction
    {

        public static npy_intp[] range(int start, int end)
        {
            npy_intp[] a = new npy_intp[end - start];

            int index = 0;
            for (int i = start; i < end; i++)
            {
                a[index] = i;
                index++;
            }

            return a;
        }

    }

}
