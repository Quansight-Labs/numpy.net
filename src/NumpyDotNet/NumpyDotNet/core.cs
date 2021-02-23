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

using NumpyLib;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet
{
    public static partial class np
    {
    
        private static ndarray ndArrayFromMD(Array ssrc, NPY_TYPES type_num, int ndim)
        {
            npy_intp []newshape = new npy_intp[ndim];
            for (int i = 0; i < ndim; i++)
            {
                newshape[i] = ssrc.GetLength(i);
            }

            dtype WantedDType = NpyCoreApi.DescrFromType(type_num);
            return np.array(new VoidPtr(ArrayFromMD(ssrc, type_num), type_num), dtype: WantedDType).reshape(newshape);
        }

        private static System.Array ArrayFromMD(Array ssrc, NPY_TYPES type_num)
        {
            var ArrayHandler = numpyAPI.GetArrayHandler(type_num);
            if (ArrayHandler != null)
            {
                return ArrayHandler.ToArray(ssrc);
            }
 
            throw new Exception("Unexpected NPY_TYPES");

        }

        public static bool IsNumericType(object o)
        {
            switch (Type.GetTypeCode(o.GetType()))
            {
                case TypeCode.Boolean:
                case TypeCode.Byte:
                case TypeCode.SByte:
                case TypeCode.UInt16:
                case TypeCode.UInt32:
                case TypeCode.UInt64:
                case TypeCode.Int16:
                case TypeCode.Int32:
                case TypeCode.Int64:
                case TypeCode.Decimal:
                case TypeCode.Double:
                case TypeCode.Single:
                    return true;
                default:
                    if (o is System.Numerics.Complex)
                        return true;
                    if (o is System.Numerics.BigInteger)
                        return true;
                    return false;
            }
        }

        public static VoidPtr GetSingleElementArray(object o)
        {
            switch (Type.GetTypeCode(o.GetType()))
            {
                case TypeCode.Boolean:
                    return new VoidPtr(new bool[] { (bool)o });
                case TypeCode.Byte:
                    return new VoidPtr(new byte[] { (byte)o });
                case TypeCode.SByte:
                    return new VoidPtr(new sbyte[] { (sbyte)o });
                case TypeCode.UInt16:
                    return new VoidPtr(new UInt16[] { (UInt16)o });
                case TypeCode.UInt32:
                    return new VoidPtr(new UInt32[] { (UInt32)o });
                case TypeCode.UInt64:
                    return new VoidPtr(new UInt64[] { (UInt64)o });
                case TypeCode.Int16:
                    return new VoidPtr(new Int16[] { (Int16)o });
                case TypeCode.Int32:
                    return new VoidPtr(new Int32[] { (Int32)o });
                case TypeCode.Int64:
                   return new VoidPtr(new Int64[] { (Int64)o });
                case TypeCode.Decimal:
                    return new VoidPtr(new Decimal[] { (Decimal)o });
                case TypeCode.Double:
                    return new VoidPtr(new Double[] { (Double)o });
                case TypeCode.Single:
                    return new VoidPtr(new Single[] { (Single)o });
                case TypeCode.String:
                    return new VoidPtr(new string[] { (string)o });
                default:
                    if (o is System.Numerics.Complex)
                    {
                        return new VoidPtr(new System.Numerics.Complex[] { (System.Numerics.Complex)o });
                    }
                    if (o is System.Numerics.BigInteger)
                    {
                        return new VoidPtr(new System.Numerics.BigInteger[] { (System.Numerics.BigInteger)o });
                    }
                    if (o is object)
                    {
                        return new VoidPtr(new object[] { (object)o });
                    }
                    throw new Exception("Unable to convert numeric type");
            }
        }

     
        private static bool hasattr(ndarray m, string v)
        {
            return true;
        }





    }
}
