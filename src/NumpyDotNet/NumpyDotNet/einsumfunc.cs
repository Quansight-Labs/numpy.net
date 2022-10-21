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
        public static (IEnumerable<object> path, string string_repr) einsum_path(string subscripts, IEnumerable<object> operands, object optimize)
        {
            string path_type = null;
            int? memory_limit = null;

            if (optimize is bool)
            {
                bool boptimize = (bool)optimize;

                if (boptimize == false)
                {
                    path_type = "None";
                }
                else
                {
                    path_type = "greedy";
                }
            }
            if (optimize is IEnumerable<string>)
            {
                IEnumerable<string> enumerable_optimize = optimize as IEnumerable<string>;
                if (enumerable_optimize != null && enumerable_optimize.Count() > 0)
                {
                    string FirstString = enumerable_optimize.ElementAt(0);
                    if (FirstString == "einsum_path")
                    {
                        path_type = "einsum_path";
                    }
                    else
                    {
                        path_type = "None";
                    }
                }

            }
            if (optimize is IEnumerable<object>)
            {
                IEnumerable<object> enumerable_optimize = optimize as IEnumerable<object>;
                if (enumerable_optimize != null && enumerable_optimize.Count() > 0)
                {
                    string FirstString = enumerable_optimize.ElementAt(0).ToString();
                    if (FirstString == "einsum_path")
                    {
                        path_type = "einsum_path";
                    }
                    else
                    {
                        path_type = "None";
                    }
                }
            }
            if (optimize is ValueTuple<string,int>)
            {
                ValueTuple<string,int> toptimize = (ValueTuple<string, int>)optimize;
                path_type = toptimize.Item1;
                memory_limit = toptimize.Item2;
            }
            if (optimize is string)
            {
                string soptimize = optimize.ToString();
                path_type = soptimize;
            }

            if (path_type == null)
            {
                throw new ValueError(string.Format("Did not understand the path"));
            }

            switch (path_type)
            {
                case "einsum_path":
                case "None":
                case "optimize":
                case "greedy":
                    break;
                default:
                    throw new ValueError(string.Format("Did not understand the path: {0}", path_type));
            }

            parse_einsum_input(subscripts, operands);

            return (null, "123");
        }

        private static (string input_string, string output_string, IEnumerable<object> operands) parse_einsum_input(string subscripts, IEnumerable<object> operands)
        {
            if (operands == null || operands.Count() == 0)
            {
                throw new ValueError("No input operands");
            }

            throw new NotImplementedException();
        }

        public static ndarray einsum()
        {
            throw new NotImplementedException();
            return null;
        }

    }
}

