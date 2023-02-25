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
        static byte[] _ZIP_PREFIX = new byte[] { 0x50, 0x4B, 0x03, 0x04 }; // b'PK\x03\x04';
        static byte[] MAGIC_PREFIX = new byte[] { 0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59 }; // b'\x93NUMPY';
        static int MAGIC_LEN = MAGIC_PREFIX.Length + 2;

        public static ndarray load(string PathName)
        {
            throw new Exception("This function is under consideration for development");

            if (string.IsNullOrEmpty(PathName))
            {
                throw new Exception("Pathname null or empty");
            }

            if (System.IO.File.Exists(PathName) == false)
            {
                throw new Exception("Specified file does not exist!");
            }

            var fp = System.IO.File.Open(PathName, System.IO.FileMode.Open);


            int N = MAGIC_PREFIX.Length;

            byte[] magic = new byte[N + 200];

            var magic_read = fp.Read(magic, 0, N);

            fp.Seek(-Math.Min(N, magic_read), System.IO.SeekOrigin.Current);

            if (IsPrefixMatch(magic, _ZIP_PREFIX))
            {
                throw new Exception("Zipped files are not supported");
            }
            if (IsPrefixMatch(magic, MAGIC_PREFIX))
            {
                return read_array(fp);
            }
            return null;
        }

        private static ndarray read_array(FileStream fp)
        {
            var version = read_magic(fp);
            _check_version(version);

            var array_info = _read_array_header(fp, version);


            throw new NotImplementedException();
        }


        private static object _read_array_header(FileStream fp, (int major, int minor) version)
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


            var hlength_str = _read_bytes(fp, struct_calcsize, "array header length");

            var header = _read_bytes(fp, hlength_str[0], "array_header");

            throw new NotImplementedException();
        }

        private static void _check_version((int major, int minor) version)
        {
            if (version.major == 1 && version.minor == 0)
                return;
            if (version.major == 2 && version.minor == 0)
                return;

            throw new Exception(string.Format("we only support version (1,0) and (2,0), not ({0},{1})", version.major, version.minor));
        }

        private static (int major, int minor) read_magic(FileStream fp)
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

        private static byte[] _read_bytes(FileStream fp, int size, string error_template)
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

        private static bool IsPrefixMatch(byte[] magic, byte[] Prefix)
        {
            for (int i = 0; i < Prefix.Length; i++)
            {
                if (magic[i] != Prefix[i])
                    return false;
            }

            return true;
        }

    }
}

