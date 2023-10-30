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
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif
namespace NumpyLib
{
    internal partial class numpyinternal
    {

        internal static NpyArray NpyArray_View(NpyArray self, NpyArray_Descr newDescr, object subtype)
        {
            NpyArray newArr = null;

            Npy_INCREF(NpyArray_DESCR(self));
            newArr = NpyArray_NewFromDescr(NpyArray_DESCR(self),
                                        NpyArray_NDIM(self), NpyArray_DIMS(self),
                                        NpyArray_STRIDES(self),
                                        NpyArray_BYTES(self), 
                                        NpyArray_FLAGS(self),
                                        false,
                                        subtype, Npy_INTERFACE(self));
            if (newArr == null)
            {
                return null;
            }

            newArr.SetBase(self);
            Npy_INCREF(self);
            Debug.Assert(null == newArr.base_obj);

            if (newDescr != null)
            {
                /* TODO: unwrap type. */
                if (NpyArray_SetDescr(newArr, newDescr) < 0)
                {
                    Npy_DECREF(newArr);
                    Npy_DECREF(newDescr);
                    return null;
                }
                Npy_DECREF(newDescr);
            }
            return newArr;
        }

        internal static int NpyArray_SetDescr(NpyArray self, NpyArray_Descr newtype)
        {
            npy_intp newdim;
            int index;
            string msg = "new type not compatible with array.";

            Npy_INCREF(newtype);

            if (newtype.elsize == 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_TypeError, "data-type must not be 0-sized");
                Npy_DECREF(newtype);
                return -1;
            }


            if ((newtype.elsize != NpyArray_ITEMSIZE(self)) &&
                (NpyArray_NDIM(self) == 0 || !NpyArray_ISONESEGMENT(self) || newtype.subarray != null))
            {
                goto fail;
            }
            if (NpyArray_ISCONTIGUOUS(self))
            {
                index = NpyArray_NDIM(self) - 1;
            }
            else
            {
                index = 0;
            }
            if (newtype.elsize < NpyArray_ITEMSIZE(self))
            {
                /*
                 * if it is compatible increase the size of the
                 * dimension at end (or at the front for FORTRAN)
                 */
                if (NpyArray_ITEMSIZE(self) % newtype.elsize != 0)
                {
                    goto fail;
                }
                newdim = (npy_intp)(NpyArray_ITEMSIZE(self) / newtype.elsize);
                newdim = newdim * NpyArray_DIM(self, index);
                NpyArray_DIM_Update(self, index, (int)newdim);
                NpyArray_STRIDE_Update(self, index, newtype.elsize);
            }
            else if (newtype.elsize > NpyArray_ITEMSIZE(self))
            {
                /*
                 * Determine if last (or first if FORTRAN) dimension
                 * is compatible
                 */
                newdim = (npy_intp)(NpyArray_DIM(self, index) * NpyArray_ITEMSIZE(self));
                if ((newdim % newtype.elsize) != 0)
                {
                    goto fail;
                }
                NpyArray_DIM_Update(self, index, (int)(newdim / newtype.elsize));
                NpyArray_STRIDE_Update(self, index,newtype.elsize);
            }

            /* fall through -- adjust type*/
            Npy_DECREF(NpyArray_DESCR(self));
            if (newtype.subarray != null)
            {
                /*
                 * create new array object from data and update
                 * dimensions, strides and descr from it
                 */
                NpyArray temp;
                /*
                 * We would decref newtype here.
                 * temp will steal a reference to it
                 */
                temp =
                    NpyArray_NewFromDescr(newtype, NpyArray_NDIM(self),
                                          NpyArray_DIMS(self), NpyArray_STRIDES(self), 
                                          NpyArray_BYTES(self), NpyArray_FLAGS(self),
                                          true, null,
                                          null);
                if (temp == null)
                {
                    return -1;
                }
                NpyDimMem_FREE(NpyArray_DIMS(self));
                NpyArray_DIMS_Update(self, NpyArray_DIMS(temp));
                NpyArray_NDIM_Update(self, NpyArray_NDIM(temp));
                NpyArray_STRIDES_Update(self,NpyArray_STRIDES(temp));
                newtype = NpyArray_DESCR(temp);
                Npy_INCREF(newtype);
                /* Fool deallocator not to delete these*/
                NpyArray_NDIM_Update(temp,0);
                NpyArray_DIMS_Update(temp,null);
                Npy_DECREF(temp);
            }

            NpyArray_DESCR_Update(self,newtype);
            NpyArray_UpdateFlags(self, NPYARRAYFLAGS.NPY_UPDATE_ALL);
            return 0;

            fail:
            NpyErr_SetString(npyexc_type.NpyExc_ValueError, msg);
            Npy_DECREF(newtype);
            return -1;
        }

        /*
        Copy an array.
        */
        internal static NpyArray NpyArray_NewCopy(NpyArray m1, NPY_ORDER order)
        {
            NpyArray ret;
            bool fortran;

            fortran = (order == NPY_ORDER.NPY_ANYORDER) ? NpyArray_ISFORTRAN(m1) : (order == NPY_ORDER.NPY_FORTRANORDER);

            Npy_INCREF(m1.descr);
            ret = NpyArray_Alloc(m1.descr, m1.nd, m1.dimensions,
                                 fortran, Npy_INTERFACE(m1));
            if (ret == null)
            {
                return null;
            }
            if (NpyArray_CopyInto(ret, m1) == -1)
            {
                Npy_DECREF(ret);
                return null;
            }

            return ret;
        }


        internal static int NpyArray_FillWithScalar(NpyArray arr, NpyArray zero_d_array)
        {
            npy_intp size;
            NpyArray from;
            VoidPtr fromptr;

 
            size = NpyArray_SIZE(arr);
            if (size == 0)
            {
                return 0;
            }

            if (!NpyArray_ISALIGNED(zero_d_array) || zero_d_array.descr.type != arr.descr.type)
            {
                Npy_INCREF(arr.descr);
                from = NpyArray_FromArray(zero_d_array, arr.descr, NPYARRAYFLAGS.NPY_ALIGNED);
                if (from == null)
                {
                    return -1;
                }
            }
            else
            {
                from = zero_d_array;
                Npy_INCREF(from);
            }

            bool swap = (NpyArray_ISNOTSWAPPED(arr) != NpyArray_ISNOTSWAPPED(from));
            fromptr = new VoidPtr(from);


            if (NpyArray_ISONESEGMENT(arr))
            {
                VoidPtr toptr = new VoidPtr(arr);
                var helper = MemCopy.GetMemcopyHelper(toptr);
                helper.FillWithScalar(toptr, fromptr, size, swap);
            }
            else
            {
                NpyArrayIterObject iter;

                iter = NpyArray_IterNew(arr);
                if (iter == null)
                {
                    Npy_DECREF(from);
                    return -1;
                }

                var helper = MemCopy.GetMemcopyHelper(iter.dataptr);
                helper.FillWithScalarIter(iter, fromptr, size, swap);
  
                Npy_DECREF(iter);
            }
            Npy_DECREF(from);
            return 0;
        }


        internal static int NpyArray_ToBinaryFile(NpyArray self, FileInfo fp)
        {
            using (var fs = fp.Create())
            {
                return NpyArray_ToBinaryStream(self, fs);
            }
        }

        internal static int NpyArray_ToBinaryStream(NpyArray self, Stream fs)
        {

            if (NpyArray_ISCONTIGUOUS(self))
            {
                if (NpyArray_WriteBinaryStream(fs, self.data, NpyArray_SIZE(self)) == false)
                {
                    string msg = string.Format("error writing array contents to file");
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, msg);
                    return -1;
                }
            }
            else
            {

                NpyArrayIterObject it = NpyArray_IterNew(self);
                while (it.index < it.size)
                {
                    if (NpyArray_WriteBinaryStream(fs, it.dataptr, NpyArray_ITEMSIZE(self)) == false)
                    {
                        string msg = string.Format("problem writing element {0} to file", it.index);
                        NpyErr_SetString(npyexc_type.NpyExc_IOError, msg);
                        Npy_DECREF(it);
                        return -1;
                    }
                    NpyArray_ITER_NEXT(it);
                }
                Npy_DECREF(it);
            }
            return 0;
        }

        internal static int NpyArray_ToTextFile(NpyArray self, FileInfo fp, string sep, string format)
        {
            using (var fs = fp.Create())
            {
                return NpyArray_ToTextStream(self, fs, sep, format);
            }
        }

        internal static int NpyArray_ToTextStream(NpyArray self, Stream fs, string sep, string format)
        {
            if (NpyArray_ISCONTIGUOUS(self))
            {
                if (NpyArray_WriteTextStream(fs, self.data, NpyArray_SIZE(self), sep, format) == false)
                {
                    string msg = string.Format("error writing array contents to file");
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, msg);
                    return -1;
                }
            }
            else
            {

                NpyArrayIterObject it = NpyArray_IterNew(self);
                while (it.index < it.size)
                {
                    if (NpyArray_WriteTextStream(fs, it.dataptr, NpyArray_ITEMSIZE(self), sep, format) == false)
                    {
                        string msg = string.Format("problem writing element {0} to file", it.index);
                        NpyErr_SetString(npyexc_type.NpyExc_IOError, msg);
                        Npy_DECREF(it);
                        return -1;
                    }
                    NpyArray_ITER_NEXT(it);
                }
                Npy_DECREF(it);
            }
            return 0;
        }
    
        private static bool NpyArray_WriteBinaryStream(Stream fs, VoidPtr vp, npy_intp dataLen)
        {
            npy_intp dataOffset = vp.data_offset;
            dataLen = dataOffset + dataLen;

            using (var binaryWriter = new BinaryWriter(fs))
            {
                switch (vp.type_num)
                {
                    default:
                        return false;

                    case NPY_TYPES.NPY_BOOL:
                        {
                            bool[] bdata = vp.datap as bool[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                binaryWriter.Write(bdata[i]);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_BYTE:
                        {
                            sbyte[] bdata = vp.datap as sbyte[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                binaryWriter.Write(bdata[i]);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_UBYTE:
                        {

                            byte[] bdata = vp.datap as byte[];

                            binaryWriter.Write(bdata, (int)dataOffset, (int)(dataLen - dataOffset));
                            //for (npy_intp i = dataOffset; i < dataLen; i++)
                            //{
                            //    binaryWriter.Write(bdata[i]);
                            //}
                            break;
                        }
                    case NPY_TYPES.NPY_INT16:
                        {
                            Int16[] bdata = vp.datap as Int16[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                binaryWriter.Write(bdata[i]);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_UINT16:
                        {
                            UInt16[] bdata = vp.datap as UInt16[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                binaryWriter.Write(bdata[i]);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_INT32:
                        {
                            Int32[] bdata = vp.datap as Int32[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                binaryWriter.Write(bdata[i]);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_UINT32:
                        {
                            UInt32[] bdata = vp.datap as UInt32[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                binaryWriter.Write(bdata[i]);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_INT64:
                        {
                            Int64[] bdata = vp.datap as Int64[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                binaryWriter.Write(bdata[i]);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_UINT64:
                        {
                            UInt64[] bdata = vp.datap as UInt64[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                binaryWriter.Write(bdata[i]);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_FLOAT:
                        {
                            float[] bdata = vp.datap as float[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                binaryWriter.Write(bdata[i]);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_DOUBLE:
                        {
                            double[] bdata = vp.datap as double[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                binaryWriter.Write(bdata[i]);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_DECIMAL:
                        {
                            decimal[] bdata = vp.datap as decimal[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                binaryWriter.Write(bdata[i]);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_COMPLEX:
                        {
                            var bdata = vp.datap as System.Numerics.Complex[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                binaryWriter.Write(bdata[i].Real);
                                binaryWriter.Write(bdata[i].Imaginary);
                            }
                            break;
                        }   

              


                }

            }
            return true;
        }

        private static bool NpyArray_WriteTextStream(Stream fs, VoidPtr vp, npy_intp dataLen, string sep, string format)
        {
            npy_intp dataOffset = vp.data_offset;
            dataLen = dataOffset + dataLen;

            using (var textwriter = new StreamWriter(fs))
            {
                StringBuilder sb = new StringBuilder();
                switch (vp.type_num)
                {
                    default:
                        return false;

                    case NPY_TYPES.NPY_BOOL:
                        {
                            if (string.IsNullOrEmpty(format))
                            {
                                format = "{0}";
                            }

                            bool[] bdata = vp.datap as bool[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                sb.AppendFormat(format, bdata[i]);
                                sb.Append(sep);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_BYTE:
                        {
                            if (string.IsNullOrEmpty(format))
                            {
                                format = "{0}";
                            }

                            sbyte[] bdata = vp.datap as sbyte[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                sb.AppendFormat(format, bdata[i]);
                                sb.Append(sep);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_UBYTE:
                        {
                            if (string.IsNullOrEmpty(format))
                            {
                                format = "{0}";
                            }

                            byte[] bdata = vp.datap as byte[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                sb.AppendFormat(format, bdata[i]);
                                sb.Append(sep);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_INT16:
                        {
                            if (string.IsNullOrEmpty(format))
                            {
                                format = "{0}";
                            }

                            Int16[] bdata = vp.datap as Int16[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                sb.AppendFormat(format, bdata[i]);
                                sb.Append(sep); textwriter.Write(bdata[i]);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_UINT16:
                        {
                            if (string.IsNullOrEmpty(format))
                            {
                                format = "{0}";
                            }

                            UInt16[] bdata = vp.datap as UInt16[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                sb.AppendFormat(format, bdata[i]);
                                sb.Append(sep);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_INT32:
                        {
                            if (string.IsNullOrEmpty(format))
                            {
                                format = "{0}";
                            }

                            Int32[] bdata = vp.datap as Int32[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                sb.AppendFormat(format, bdata[i]);
                                sb.Append(sep);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_UINT32:
                        {
                            if (string.IsNullOrEmpty(format))
                            {
                                format = "{0}";
                            }

                            UInt32[] bdata = vp.datap as UInt32[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                sb.AppendFormat(format, bdata[i]);
                                sb.Append(sep);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_INT64:
                        {
                            if (string.IsNullOrEmpty(format))
                            {
                                format = "{0}";
                            }

                            Int64[] bdata = vp.datap as Int64[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                sb.AppendFormat(format, bdata[i]);
                                sb.Append(sep);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_UINT64:
                        {
                            if (string.IsNullOrEmpty(format))
                            {
                                format = "{0}";
                            }

                            UInt64[] bdata = vp.datap as UInt64[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                sb.AppendFormat(format, bdata[i]);
                                sb.Append(sep);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_FLOAT:
                        {
                            if (string.IsNullOrEmpty(format))
                            {
                                format = "{0:0.0#######}";
                            }
                            float[] bdata = vp.datap as float[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                sb.AppendFormat(format, bdata[i]);
                                sb.Append(sep);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_DOUBLE:
                        {
                            if (string.IsNullOrEmpty(format))
                            {
                                format = "{0:0.0###############}";
                            }
                            double[] bdata = vp.datap as double[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                sb.AppendFormat(format, bdata[i]);
                                sb.Append(sep);
                            }
                            break;
                        }
                    case NPY_TYPES.NPY_DECIMAL:
                        {
                            if (string.IsNullOrEmpty(format))
                            {
                                format = "{0:0.0###############}";
                            }

                            decimal[] bdata = vp.datap as decimal[];
                            for (npy_intp i = dataOffset; i < dataLen; i++)
                            {
                                sb.AppendFormat(format, bdata[i]);
                                sb.Append(sep);
                            }
                            break;
                        }

                }

                // remove the last seperator and write to the file
                sb.Remove(sb.Length - sep.Length, sep.Length);
                textwriter.Write(sb.ToString());

            }

 
            return true;
        }

        internal static NpyArray NpyArray_FromTextFile(FileInfo fp, NpyArray_Descr descr, npy_intp num, string sep)
        {
            using (var fs = fp.OpenRead())
            {
                return NpyArray_FromTextStream(fs, descr, num, sep);
            }
        }

        internal static NpyArray NpyArray_FromString(string data, npy_intp slen, NpyArray_Descr descr, npy_intp num, string sep)
        {
            using (var fs = new MemoryStream(Encoding.UTF8.GetBytes(data)))
            {
                return NpyArray_FromTextStream(fs, descr, num, sep);
            }
        }

        internal static NpyArray NpyArray_FromTextStream(Stream fileStream, NpyArray_Descr descr, npy_intp num, string sep)
        {

            if (descr.elsize == 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "The elements are 0-sized.");
                Npy_DECREF(descr);
                return null;
            }
            if (string.IsNullOrEmpty(sep))
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "A separator must be specified when reading a text file.");
                Npy_DECREF(descr);
                return null;
            }
  
            List<string> entries = new List<string>();
            using (StreamReader sr = new StreamReader(fileStream))
            {
                string buffer = "";
                while (sr.Peek() >= 0)
                {
                    buffer += (char)sr.Read();
                    if (buffer.EndsWith(sep))
                    {
                        entries.Add(buffer.Substring(0, buffer.IndexOf(sep)));
                        buffer = "";
                    }
                    if (num > 0 && num <= entries.Count)
                        break;
                }
                if (!string.IsNullOrEmpty(buffer))
                {
                    entries.Add(buffer);
                }
            }

            var data = NpyDataMem_NEW(descr.type_num, (ulong)entries.Count, false);

            for (int i = 0; i < entries.Count; i++)
            {
                SetIndex(data, i, entries[i]);
            }

            npy_intp[] dims = new npy_intp[] { entries.Count };

            NpyArray newArr = NpyArray_NewFromDescr(descr, 1, dims, null, data, NPYARRAYFLAGS.NPY_CARRAY, false, null, null);
            return newArr;

        }

        internal static NpyArray NpyArray_FromBinaryFile(FileInfo fp, NpyArray_Descr descr, npy_intp num)
        {
            using (var fs = fp.OpenRead())
            {
                return NpyArray_FromBinaryStream(fs, descr, num);
            }
        }

        internal static NpyArray NpyArray_FromBinaryStream(Stream fileStream, NpyArray_Descr descr, npy_intp num)
        {
 
            if (descr.elsize == 0)
            {
                NpyErr_SetString(npyexc_type.NpyExc_ValueError, "The elements are 0-sized.");
                Npy_DECREF(descr);
                return null;
            }

            int index = 0;
            var data = NpyDataMem_NEW(descr.type_num, (ulong)((fileStream.Length - fileStream.Position)/ descr.elsize), false);
            using (BinaryReader sr = new BinaryReader(fileStream))
            {
                while (true)
                {
                    try
                    {
                        switch (descr.type_num)
                        {
                            case NPY_TYPES.NPY_BOOL:
                                {
                                    bool[] bdata = data.datap as bool[];
                                    bdata[index] = sr.ReadBoolean();
                                    index++;
                                    break;
                                }
   
                            case NPY_TYPES.NPY_BYTE:
                                {
                                    sbyte[] bdata = data.datap as sbyte[];
                                    bdata[index] = sr.ReadSByte();
                                    index++;
                                    break;
                                }
                            case NPY_TYPES.NPY_UBYTE:
                                {
                                    byte[] bdata = data.datap as byte[];
                                    bdata[index] = sr.ReadByte();
                                    index++;
                                    break;
                                }
                            case NPY_TYPES.NPY_INT16:
                                {
                                    Int16[] bdata = data.datap as Int16[];
                                    bdata[index] = sr.ReadInt16();
                                    index++;
                                    break;
                                }
                            case NPY_TYPES.NPY_UINT16:
                                {
                                    UInt16[] bdata = data.datap as UInt16[];
                                    bdata[index] = sr.ReadUInt16();
                                    index++;
                                    break;
                                }
                            case NPY_TYPES.NPY_INT32:
                                {
                                    Int32[] bdata = data.datap as Int32[];
                                    bdata[index] = sr.ReadInt32();
                                    index++;
                                    break;
                                }
                            case NPY_TYPES.NPY_UINT32:
                                {
                                    UInt32[] bdata = data.datap as UInt32[];
                                    bdata[index] = sr.ReadUInt32();
                                    index++;
                                    break;
                                }
                            case NPY_TYPES.NPY_INT64:
                                {
                                    Int64[] bdata = data.datap as Int64[];
                                    bdata[index] = sr.ReadInt64();
                                    index++;
                                    break;
                                }
                            case NPY_TYPES.NPY_UINT64:
                                {
                                    UInt64[] bdata = data.datap as UInt64[];
                                    bdata[index] = sr.ReadUInt32();
                                    index++;
                                    break;
                                }
                            case NPY_TYPES.NPY_FLOAT:
                                {
                                    float[] bdata = data.datap as float[];
                                    bdata[index] = sr.ReadSingle();
                                    index++;
                                    break;
                                }
                            case NPY_TYPES.NPY_DOUBLE:
                                {
                                    double[] bdata = data.datap as double[];
                                    bdata[index] = sr.ReadDouble();
                                    index++;
                                    break;
                                }
                            case NPY_TYPES.NPY_DECIMAL:
                                {
                                    decimal[] bdata = data.datap as decimal[];
                                    bdata[index] = sr.ReadDecimal();
                                    index++;
                                    break;
                                }

                            case NPY_TYPES.NPY_COMPLEX:
                                {
                                    System.Numerics.Complex[] bdata = data.datap as System.Numerics.Complex[];
                                    bdata[index] = new System.Numerics.Complex(sr.ReadDouble(),sr.ReadDouble());
                                    index++;
                                    break;
                                }

                            default:
                            case NPY_TYPES.NPY_BIGINT:
                            case NPY_TYPES.NPY_STRING:
                            case NPY_TYPES.NPY_OBJECT:
                                {
                                    throw new Exception("This function does not support this dtype");
                                }

                        }
                    }
                    catch (Exception ex)
                    {
                        break;
                    }

                    if (num > 0 && num <= index)
                        break;
                }
    
       
            }

 
            npy_intp[] dims = new npy_intp[] { index };

            NpyArray newArr = NpyArray_NewFromDescr(descr, 1, dims, null, data, NPYARRAYFLAGS.NPY_CARRAY, false, null, null);
            return newArr;

        }

 
    }
}
