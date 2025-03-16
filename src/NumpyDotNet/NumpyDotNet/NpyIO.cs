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
using System.Linq;
using System.Numerics;
using System.Text;
using System.Runtime.InteropServices;
using NumpyLib;
using System.IO;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet
{
    /// <summary>
    /// implements functions similar to npyio.py
    /// </summary>
    public static partial class np
    {
        private class FileLoadHeader
        {
            public bool success;
            public NPY_TYPES NpyType;
            public int itemsize;
            public char byteorder;
            public bool fortran_order;
            public shape shape;
        }


        static byte[] _ZIP_PREFIX = new byte[] { 0x50, 0x4B, 0x03, 0x04 }; // b'PK\x03\x04';
        static byte[] MAGIC_PREFIX = new byte[] { 0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59 }; // b'\x93NUMPY';
        static int MAGIC_LEN = MAGIC_PREFIX.Length + 2;

        #region np.load

        /// <summary>
        /// Load arrays or pickled objects from .npy files
        /// </summary>
        /// <param name="PathName">complete file name to load into an ndarray</param>
        /// <returns></returns>
        public static ndarray load(string PathName)
        {
            if (string.IsNullOrEmpty(PathName))
            {
                throw new Exception("Pathname null or empty");
            }

            if (System.IO.File.Exists(PathName) == false)
            {
                throw new Exception("Specified file does not exist!");
            }

            // open the specified file
            var fp = System.IO.File.Open(PathName, System.IO.FileMode.Open);
            BinaryReader reader = new BinaryReader(fp, Encoding.Default, false);

            int N = MAGIC_PREFIX.Length;

            byte[] magic = new byte[N + 200];

            // read in enough to get the magic header
            var magic_read =  reader.Read(magic, 0, N);

            // reset the stream pointer to after magic header
            fp.Seek(-Math.Min(N, magic_read), System.IO.SeekOrigin.Current);

            // if the header indicates zip file, throw exceptipn
            if (IsPrefixMatch(magic, _ZIP_PREFIX))
            {
                throw new Exception("Zipped files are not supported");
            }

            // if the header matches the expected, ready the array.
            if (IsPrefixMatch(magic, MAGIC_PREFIX))
            {
                return read_array(reader);
            }
            return null;
        }

        private static ndarray read_array(BinaryReader br)
        {
            // read the magic header and check the version number
            var version = read_magic(br);
            _check_version(version);

            // read and parse the embedded header
            var array_info = _read_array_header(br, version);
            if (array_info.success == false)
            {
                throw new Exception("unable to read the file format");
            }

            if ((array_info.byteorder == '<' && !BitConverter.IsLittleEndian) ||
                (array_info.byteorder == '>' && BitConverter.IsLittleEndian))
                throw new NotSupportedException("Byte order doesn't match system endianness");


            NPY_ORDER NpyOrder = NPY_ORDER.NPY_CORDER;
            if (array_info.fortran_order == true)
                NpyOrder = NPY_ORDER.NPY_FORTRANORDER;


            // calculate the number of elements from the embedded shape
            int length = (int)CalculateNewShapeSize(array_info.shape);

            // read the rest of the file as bytes.
            byte[] buffer = br.ReadBytes(array_info.itemsize * length);


            // recreate the original array with the information embedded in the header
            switch (array_info.NpyType)
            {
                case NPY_TYPES.NPY_BOOL:
                    {
                        Array array = Array.CreateInstance(typeof(System.Boolean), length);
                        Buffer.BlockCopy(buffer, 0, array, 0, buffer.Length);
                        return np.array((bool[])array).reshape(array_info.shape, order: NpyOrder);
                    }
                case NPY_TYPES.NPY_BYTE:
                    {
                        Array array = Array.CreateInstance(typeof(System.SByte), length);
                        Buffer.BlockCopy(buffer, 0, array, 0, buffer.Length);
                        return np.array((sbyte[])array).reshape(array_info.shape, order: NpyOrder);
                    }
                case NPY_TYPES.NPY_UBYTE:
                    {
                        Array array = Array.CreateInstance(typeof(System.Byte), length);
                        Buffer.BlockCopy(buffer, 0, array, 0, buffer.Length);
                        return np.array((byte[])array).reshape(array_info.shape, order: NpyOrder);
                    }
                case NPY_TYPES.NPY_INT16:
                    {
                        Array array = Array.CreateInstance(typeof(System.Int16), length);
                        Buffer.BlockCopy(buffer, 0, array, 0, buffer.Length);
                        return np.array((Int16[])array).reshape(array_info.shape, order: NpyOrder);
                    }
                case NPY_TYPES.NPY_UINT16:
                    {
                        Array array = Array.CreateInstance(typeof(System.UInt16), length);
                        Buffer.BlockCopy(buffer, 0, array, 0, buffer.Length);
                        return np.array((UInt16[])array).reshape(array_info.shape, order: NpyOrder);
                    }
                case NPY_TYPES.NPY_INT32:
                    {
                        Array array = Array.CreateInstance(typeof(System.Int32), length);
                        Buffer.BlockCopy(buffer, 0, array, 0, buffer.Length);
                        return np.array((Int32[])array).reshape(array_info.shape, order: NpyOrder);
                    }
                case NPY_TYPES.NPY_UINT32:
                    {
                        Array array = Array.CreateInstance(typeof(System.UInt32), length);
                        Buffer.BlockCopy(buffer, 0, array, 0, buffer.Length);
                        return np.array((UInt32[])array).reshape(array_info.shape, order: NpyOrder);
                    }
                case NPY_TYPES.NPY_INT64:
                    {
                        Array array = Array.CreateInstance(typeof(System.Int64), length);
                        Buffer.BlockCopy(buffer, 0, array, 0, buffer.Length);
                        return np.array((Int64[])array).reshape(array_info.shape, order: NpyOrder);
                    }
                case NPY_TYPES.NPY_UINT64:
                    {
                        Array array = Array.CreateInstance(typeof(System.UInt64), length);
                        Buffer.BlockCopy(buffer, 0, array, 0, buffer.Length);
                        return np.array((UInt64[])array).reshape(array_info.shape, order: NpyOrder);
                    }
                case NPY_TYPES.NPY_FLOAT:
                    {
                        Array array = Array.CreateInstance(typeof(System.Single), length);
                        Buffer.BlockCopy(buffer, 0, array, 0, buffer.Length);
                        return np.array((Single[])array).reshape(array_info.shape, order: NpyOrder);
                    }
                case NPY_TYPES.NPY_DOUBLE:
                    {
                        Array array = Array.CreateInstance(typeof(System.Double), length);
                        Buffer.BlockCopy(buffer, 0, array, 0, buffer.Length);
                        return np.array((double[])array).reshape(array_info.shape, order: NpyOrder);
                    }

                case NPY_TYPES.NPY_DECIMAL:
                    {
                        int[] bits = new int[4];

                        int Length = buffer.Length / sizeof(decimal);
                        decimal[] arr = new decimal[Length];
                        for (int i = 0; i < Length; i++)
                        {
                            int j = i * sizeof(decimal);

                            bits[0] = BitConverter.ToInt32(buffer, i * sizeof(decimal));

                            arr[i] = new decimal(bits);
                        }

                        return np.array(arr).reshape(array_info.shape, order: NpyOrder);
                    }
                case NPY_TYPES.NPY_COMPLEX:
                    {
                        Array array = Array.CreateInstance(typeof(System.Double), length*2);
                        Buffer.BlockCopy(buffer, 0, array, 0, buffer.Length);
                        double[] darray = (double[])array;

                        int Length = buffer.Length / (sizeof(double) * 2);
                        Complex[] arr = new Complex[Length];

                        for (int i = 0, j = 0; i < darray.Length; i+= 2, j++)
                        {
                            double Real = darray[i];
                            double Imag = darray[i + 1];

                            arr[j] = new Complex(Real, Imag);
                        }

                        return np.array(arr).reshape(array_info.shape, order: NpyOrder);
                    }

                default:
                    throw new Exception("unsupported data type");
            }


            throw new Exception("unable to read the file format");

        }

    
        // read the header information in the specified file
        private static FileLoadHeader _read_array_header(BinaryReader fp, (int major, int minor) version)
        {
            int struct_calcsize = -1;
            if (version.major == 1 && version.minor == 0)
            {
                struct_calcsize = 2;
            }
            else if (version.major == 2 && version.minor == 0)
            {
                struct_calcsize = 4;
            }
            else
            {
                throw new ValueError(string.Format("Invalid version ({0},{1})" + version.major, version.minor));
            }

            // skip over the version information.  We already have it.
            var hlength_str = _read_bytes(fp, struct_calcsize, "array header length");

            // read the array header in and convert it to ASCII string
            var header = _read_bytes(fp, hlength_str[0], "array_header");
            string headerStr = Encoding.ASCII.GetString(header);

            // seperate each of the array header segments.
            string descrString = HeaderSegment(headerStr, "descr", ",");
            string fortranOrderString = HeaderSegment(headerStr, "fortran_order", ",");
            string shapeString = HeaderSegment(headerStr, "shape", ")");


            // decode each of the seperated header segments.
            FileLoadHeader headerInfo = new FileLoadHeader();

            (headerInfo.success, headerInfo.NpyType, headerInfo.byteorder, headerInfo.itemsize) = ParseDescrString(descrString);
            headerInfo.fortran_order = ParseFortranOrder(fortranOrderString);
            headerInfo.shape = ParseShape(shapeString);

            // if shape is null, this is a bad parse.
            if (headerInfo.shape == null)
                headerInfo.success = false;


            return headerInfo;
        }


        // gets the substring from the startingString to the endingString
        private static string HeaderSegment(string headerStr, string startingString, string endingString)
        {
            int startIndex = headerStr.IndexOf(startingString);
            int endIndex = headerStr.IndexOf(endingString, startIndex);

            string segment = headerStr.Substring(startIndex, endIndex - startIndex+1);

            return segment;
        }

     
        //parse the array information from the description string
        private static (bool, NPY_TYPES, char, int) ParseDescrString(string descr)
        {
            string[] descrParts = descr.Split(':');
            if (descrParts.Length != 2)
                return (false, NPY_TYPES.NPY_NOTSET, '\0', 0);

            string typestr = descrParts[1];
            typestr = typestr.Replace("'", "").Trim();
            typestr = typestr.Replace(",", "").Trim();

            if (typestr is null)
                return (false, NPY_TYPES.NPY_NOTSET, '\0', 0);

            if (typestr.Length == 0)
                return (false, NPY_TYPES.NPY_NOTSET, '\0', 0);

            char byteorder;

            switch (typestr[0])
            {
                case '>':
                case '<':
                case '=':
                    byteorder = typestr[0];
                    typestr = skip1char(typestr);
                    break;
                case '|':
                    byteorder = '=';
                    typestr = skip1char(typestr);
                    break;
                default:
                    byteorder = '=';
                    break;
            }

            if (typestr.Length == 0)
                return (false, NPY_TYPES.NPY_NOTSET, '\0', 0);


            bool success = true;
            NPY_TYPES dataType = NPY_TYPES.NPY_OBJECT;
            int itemsize = 0;

            switch (typestr)
            {
                case "b1":
                    dataType = NPY_TYPES.NPY_BOOL;
                    itemsize = 1;
                    break;
                case "i1":
                    dataType = NPY_TYPES.NPY_BYTE;
                    itemsize = 1;
                    break;
                case "i2":
                    dataType = NPY_TYPES.NPY_INT16;
                    itemsize = 2;
                    break;
                case "i4":
                    dataType = NPY_TYPES.NPY_INT32;
                    itemsize = 4;
                    break;
                case "i8":
                    dataType = NPY_TYPES.NPY_INT64;
                    itemsize = 8;
                    break;
                case "u1":
                    dataType = NPY_TYPES.NPY_UBYTE;
                    itemsize = 1;
                    break;
                case "u2":
                    dataType = NPY_TYPES.NPY_UINT16;
                    itemsize = 2;
                    break;
                case "u4":
                    dataType = NPY_TYPES.NPY_UINT32;
                    itemsize = 4;
                    break;
                case "u8":
                    dataType = NPY_TYPES.NPY_UINT64;
                    itemsize = 8;
                    break;
                case "f4":
                    dataType = NPY_TYPES.NPY_FLOAT;
                    itemsize = 4;
                    break;
                case "f8":
                    dataType = NPY_TYPES.NPY_DOUBLE;
                    itemsize = 8;
                    break;
                case "d8":
                    dataType = NPY_TYPES.NPY_DECIMAL;
                    itemsize = sizeof(decimal);
                    break;
                case "cc":
                    dataType = NPY_TYPES.NPY_COMPLEX;
                    itemsize = sizeof(double)*2;
                    break;
                default:
                    success = false;
                    dataType = NPY_TYPES.NPY_OBJECT;
                    break;

            }

            return (success, dataType, byteorder, itemsize);

        }

        //parse the shape embedded in the file header
        private static shape ParseShape(string shapeString)
        {
            string[] shapeParts = shapeString.Split(':');
            if (shapeParts.Length != 2)
                return null;

            string shapeStr = shapeParts[1];

            shapeStr = shapeStr.Replace("(", "");
            shapeStr = shapeStr.Replace(")", "");
            shapeStr = shapeStr.Replace("L", "");

            shapeParts = shapeStr.Split(',');

            int[] ishape = ConvertStringToInt(shapeParts);

            return new NumpyDotNet.shape(ishape);
        }

        //parse the fortran_order embedded in the file header
        private static bool ParseFortranOrder(string fortranOrder)
        {
            string[] fortranParts = fortranOrder.Split(':');
            if (fortranParts.Length != 2)
                return false;

            if (fortranParts[1].ToLower().Contains("true"))
                return true;
            else
                return false;
        }

        // converts the header shape field into a numeric shape
        private static int[] ConvertStringToInt(string[] shapeParts)
        {
            List<int> intParts = new List<int>();

            for (int i = 0; i < shapeParts.Length; i++)
            {
                if (!string.IsNullOrEmpty(shapeParts[i]))
                    intParts.Add(int.Parse(shapeParts[i]));
            }

            return intParts.ToArray();
        }

        // skips the input string ahead by 1 char
        private static string skip1char(string typestr)
        {
            return typestr.Substring(1);
        }

        // validates the version information from the file
        private static void _check_version((int major, int minor) version)
        {
            if (version.major == 1 && version.minor == 0)
                return;
            if (version.major == 2 && version.minor == 0)
                return;

            throw new Exception(string.Format("we only support version (1,0) and (2,0), not ({0},{1})", version.major, version.minor));
        }

        // reads the magic header field
        private static (int major, int minor) read_magic(BinaryReader fp)
        {
            byte[] magic_str = _read_bytes(fp, MAGIC_LEN, "magic string");
            if (IsPrefixMatch(magic_str, MAGIC_PREFIX) == false)
            {
                throw new Exception("File does not have expected magic string");
            }

            int major = magic_str[MAGIC_LEN - 2];
            int minor = magic_str[MAGIC_LEN - 1];

            return (major, minor);
        }

        // reads the specified number of bytes
        private static byte[] _read_bytes(BinaryReader fp, int size, string error_template)
        {

            byte[] data = new byte[size];
            int data_offset = 0;

            while (true)
            {
                try
                {
                    int r = fp.Read(data, data_offset, size - data_offset);
                    data_offset += r;
                    if (data_offset == size)
                        break;
                }
                catch (Exception ex)
                {
                    string msg = string.Format("EOF: reading {0}, expected {1} bytes got {2}", error_template, size, data_offset);
                    throw new Exception(msg);
                }
            }

            return data;
        }

        // compares the header prefix with expected sequence
        private static bool IsPrefixMatch(byte[] magic, byte[] Prefix)
        {
            for (int i = 0; i < Prefix.Length; i++)
            {
                if (magic[i] != Prefix[i])
                    return false;
            }

            return true;
        }

        #endregion

        #region np.save

        /// <summary>
        /// Save an array to a binary file in NumPy .npy format.
        /// </summary>
        /// <param name="PathName">pathname to save file</param>
        /// <param name="array">array to save as binary file</param>
        public static void save(string PathName, ndarray array)
        {
            var fp = System.IO.File.Open(PathName, System.IO.FileMode.Create);
            BinaryWriter writer = new BinaryWriter(fp, Encoding.Default, false);

            writer.Write(MAGIC_PREFIX);
            writer.Write((byte)1);  // major version
            writer.Write((byte)0);  // minor version

            string Header = CreateHeader(array);
            byte[] headerbytes = Encoding.ASCII.GetBytes(Header);

            writer.Write((short)headerbytes.Length);
            writer.Write(headerbytes); // header


            if (array.TypeNum == NPY_TYPES.NPY_DECIMAL)
            {
                decimal[] darray = array.rawdata(0).datap as decimal[];
                foreach (decimal d in darray)
                {
                    writer.Write(d);
                }
            }
            else
            if (array.TypeNum == NPY_TYPES.NPY_COMPLEX)
            {
                Complex[] carray = array.rawdata(0).datap as Complex[];
                foreach (Complex d in carray)
                {
                    writer.Write(d.Real);
                    writer.Write(d.Imaginary);
                }
            }
            else
            {
                byte[] rawData = array.tobytes();
                writer.Write(rawData);
            }
   


            writer.Close();
            fp.Close();


        }

        private static string CreateHeader(ndarray array)
        {
            string byteorder = BitConverter.IsLittleEndian ? "<" : ">";
            string TypeIndicator = "";

            switch (array.TypeNum)
            {
                case NPY_TYPES.NPY_BOOL:
                    TypeIndicator = "b1";
                    break;
                case NPY_TYPES.NPY_BYTE:
                    TypeIndicator = "i1";
                    break;
                case NPY_TYPES.NPY_INT16:
                    TypeIndicator = "i2";
                    break;
                case NPY_TYPES.NPY_INT32:
                    TypeIndicator = "i4";
                    break;
                case NPY_TYPES.NPY_INT64:
                    TypeIndicator = "i8";
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    TypeIndicator = "u1";
                    break;
                case NPY_TYPES.NPY_UINT16:
                    TypeIndicator = "u2";
                    break;
                case NPY_TYPES.NPY_UINT32:
                    TypeIndicator = "u4";
                    break;
                case NPY_TYPES.NPY_UINT64:
                    TypeIndicator = "u8";
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    TypeIndicator = "f4";
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    TypeIndicator = "f8";
                    break;
                case NPY_TYPES.NPY_DECIMAL:
                    TypeIndicator = "d8";
                    break;
                case NPY_TYPES.NPY_COMPLEX:
                    TypeIndicator = "cc";
                    break;
                default:
                    throw new Exception("Unsupported data type for this operation");
            }


            string descr = byteorder + TypeIndicator;
            string shapestr = CreateShapeString(array.shape);

            return $@"{{'descr': '{descr}', 'fortran_order': False, 'shape':({shapestr}),}}";

        }

        private static string CreateShapeString(shape shape)
        {
            string shapeString = "";
            foreach (var x in shape.iDims)
            {
                shapeString += x.ToString() + "L,";
            }

            return shapeString;
        }
        #endregion
    }
}

