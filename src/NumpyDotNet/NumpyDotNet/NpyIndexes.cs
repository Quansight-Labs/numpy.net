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
using System.Text;
using System.Runtime.InteropServices;
using NumpyLib;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif

namespace NumpyDotNet
{

    internal class NpyIndexes
    {

        public NpyIndexes()
        {
            indexes = new NpyIndex[NpyDefs.NPY_MAXDIMS];
            for (int i = 0; i < indexes.Length; i++)
            {
                indexes[i] = new NpyIndex();
            }
            num_indexes = 0;
            num_newindexes = 0;
        }

        ~NpyIndexes()
        {
            FreeIndexes();
        }

        private void FreeIndexes()
        {
            if (indexes != null) {
                if (num_indexes > 0)
                {
                    num_indexes = 0;
                }
                indexes = null;
            }
            if (strings != null)
            {
                strings = null;
            }
        }

        public int NumIndexes
        {
            get
            {
                return num_indexes;
            }
        }

        public NpyIndex[]Indexes
        {
            get
            {
                return indexes;
            }
        }

   
        /// <summary>
        /// Whether or not this is a simple (not fancy) index.
        /// </summary>
        public bool IsSimple
        {
            get
            {
                for (int i = 0; i < num_indexes; i++)
                {
                    switch (IndexType(i))
                    {
                        case NpyIndexType.NPY_INDEX_BOOL_ARRAY:
                        case NpyIndexType.NPY_INDEX_INTP_ARRAY:
                        case NpyIndexType.NPY_INDEX_STRING:
                            return false;
                    }
                }
                return true;
            }
        }

        /// <summary>
        /// Returns true if this index is all strings accessing fields in the array.
        /// </summary>
        public bool IsMultiField {
            get {
                for (int i = 0; i < num_indexes; i++) {
                    if (IndexType(i) != NpyIndexType.NPY_INDEX_STRING) {
                        return false;
                    }
                }
                return true;
            }
        }

        /// <summary>
        /// Returns whether or not this index is a single item index for an array on size ndims.
        /// </summary>
        public bool IsSingleItem(int ndims)
        {
            if (num_indexes != ndims)
            {
                return false;
            }
            for (int i = 0; i < num_indexes; i++)
            {
                if (IndexType(i) != NpyIndexType.NPY_INDEX_INTP)
                {
                    return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Finds the offset for a single item assignment to the array.
        /// </summary>
        /// <param name="arr">The array we are assigning to.</param>
        /// <returns>The offset or -1 if this is not a single assignment.</returns>
        public npy_intp SingleAssignOffset(ndarray arr)
        {
            // Check to see that there are just newaxis, ellipsis, intp or bool indexes
            for (int i = 0; i < num_indexes; i++)
            {
                switch (IndexType(i))
                {
                    case NpyIndexType.NPY_INDEX_NEWAXIS:
                    case NpyIndexType.NPY_INDEX_ELLIPSIS:
                    case NpyIndexType.NPY_INDEX_INTP:
                    case NpyIndexType.NPY_INDEX_BOOL:
                        break;
                    default:
                        return -1;
                }
            }

            // Bind to the array and calculate the offset.
            NpyIndexes bound = Bind(arr);
            {
                npy_intp offset = 0;
                int nd = 0;

                for (int i = 0; i < bound.num_indexes; i++)
                {
                    switch (bound.IndexType(i))
                    {
                        case NpyIndexType.NPY_INDEX_NEWAXIS:
                            break;
                        case NpyIndexType.NPY_INDEX_INTP:
                            offset += arr.Stride(nd++) * bound.GetIntP(i);
                            break;
                        case NpyIndexType.NPY_INDEX_SLICE:
                            // An ellipsis became a slice on binding.
                            // This is not a single item assignment.
                            return -1;
                        default:
                            // This should never happen
                            return -1;
                    }
                }
                if (nd != arr.ndim)
                {
                    // Not enough indexes. This is not a single item.
                    return -1;
                }
                return offset;
            }
        }

        public NpyIndexes Bind(ndarray arr)
        {
            NpyIndexes result = new NpyIndexes();
            int n = NpyCoreApi.BindIndex(arr, this, result);
            if (n < 0)
            {
                NpyCoreApi.CheckError();
            }
            else
            {
                result.num_indexes = n;
            }
            return result;
        }


        public void AddCSharpTuple(CSharpTuple CSTuple)
        {
            IsAdvancedIndexing = true;
            indexes[num_indexes].type = NpyIndexType.NPY_INDEX_SLICE;

            NpyIndexSlice IndexSlice = new NpyIndexSlice()
            {
                start = (npy_intp)CSTuple.index1,
                stop = CSTuple.index2 != null ? (npy_intp)CSTuple.index2+1 : (npy_intp)CSTuple.index1+1,
                step = CSTuple.index3 != null ? (npy_intp)CSTuple.index3 : 1,
            };

            indexes[num_indexes].slice = IndexSlice;

            ++num_indexes;
        }

        public void AddIndex(bool value)
        {
            // Write the type
            indexes[num_indexes].type = NpyIndexType.NPY_INDEX_BOOL;

            // Write the data
            indexes[num_indexes].boolean = value;
            ++num_indexes;
        }

        public void AddIndex(npy_intp value)
        {
            // Write the type
            indexes[num_indexes].type = NpyIndexType.NPY_INDEX_INTP;

            // Write the data
            indexes[num_indexes].intp = (npy_intp)value;
 
            ++num_indexes;
        }

        public void AddIndex(ndarray arr, string s, int index)
        {
            index -= num_newindexes;
            if (arr != null)
            {
                Slice Slice = ParseIndexString(arr, index, s, false);
                if (Slice != null)
                {
                    AddIndex(Slice);
                    return;
                }

                CSharpTuple CSharpTuple = ParseCSharpTupleString(arr, index, s, false);
                if (CSharpTuple != null)
                {
                    AddCSharpTuple(CSharpTuple);
                    return;
                }

            }
            // Write the type
            indexes[num_indexes].type = NpyIndexType.NPY_INDEX_STRING;

            // Write the data
            indexes[num_indexes]._string = s;
    
            // Save the string
            if (strings == null)
            {
                strings = new List<string>();
            }
            strings.Add(s);

            ++num_indexes;
        }

        private CSharpTuple ParseCSharpTupleString(ndarray arr, int index, string s, bool UseLiteralRanges)
        {
            try
            {
                if (s.StartsWith("[") && s.EndsWith("]"))
                {
                    string indexstring = s.Trim('[').Trim(']');

                    string[] indexParts = indexstring.Split(',');
                    if (indexParts.Length == 1)
                    {
                        return new CSharpTuple(int.Parse(indexParts[0]));
                    }
                    if (indexParts.Length == 2)
                    {
                        return new CSharpTuple(int.Parse(indexParts[0]), int.Parse(indexParts[1]));
                    }
                    if (indexParts.Length == 3)
                    {
                        return new CSharpTuple(int.Parse(indexParts[0]), int.Parse(indexParts[1]), int.Parse(indexParts[2]));
                    }
                }
            }
            catch { }

            return null;
        }

        private Slice ParseIndexString(ndarray arr, int index, object s, bool UseLiteralRanges)
        {
            npy_intp startingIndex = 0;
            npy_intp endingIndex = 0;
            npy_intp step = 0;

            //https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
            if (s is int)
            {
                int? i = s as int?;

                if (!i.HasValue)
                {
                    return null;
                }
                startingIndex = i.Value;
                if (UseLiteralRanges)
                {
                    endingIndex = startingIndex + 1;
                }
                else
                {
                    endingIndex = arr.Dim(index);
                }
                step = 1;
            }
            else
            if (s is string)
            {
                string ss = s as String;
                npy_intp i = 0;
                npy_intp j = arr.Dim(index);
                npy_intp k = 1;

                // check if this is a CSharpTuple 
                if (ss.StartsWith("[") && ss.EndsWith("]"))
                {
                    return null;
                }

                string[] Parts = ss.Split(':');
                if (Parts.Length > 3)
                {
                    return null;
                }

                if (Parts.Length == 3)
                {
                    int kt;
                    if (int.TryParse(Parts[2], out kt))
                    {
                        k = kt;
                        if (k < 0)
                        {
                            i = arr.Dim(index) - 1;
                            j = -arr.Dim(index) - 1;
                        }
                        else
                        {
                            i = 0;
                            j = arr.Dim(index);
                        }
                    }

                }
                if (Parts.Length >= 2)
                {
                    int jt;
                    if (int.TryParse(Parts[1], out jt))
                    {
                        if (jt < 0)
                        {
                            j = arr.Dim(index) + jt;
                        }
                        else
                        {
                            j = jt;
                        }
                    }
                }
                if (Parts.Length >= 1)
                {
                    int it;
                    if (int.TryParse(Parts[0], out it))
                    {
                        if (it < 0)
                        {
                            i = arr.Dim(index) + it;
                        }
                        else
                        {
                            i = it;
                        }
                        if (Parts.Length == 1 && UseLiteralRanges)
                        {
                            j = i + 1;
                        }
                    }
                }

                if (ss == ":" || ss == "::")
                {
                    i = 0;
                    j = arr.Dim(index);
                    k = 1;
                }

                startingIndex = i;
                endingIndex = j;
                step = k;

            }

            return new Slice(startingIndex, endingIndex, step);

        }

        npy_intp ConvertSliceValue(object val)
        {
            #if NPY_INTP_64
            return Convert.ToInt64(val);
            #else
            return Convert.ToInt32(val);
            #endif
        }

        public void AddIndex(ISlice slice)
        {
            npy_intp step;
            bool negativeStep;
            npy_intp start;
            npy_intp stop;
            bool hasStop;

            // Find the step
            if (slice.Step == null)
            {
                step = (npy_intp)1;
                negativeStep = false;
            }
            else
            {
                step = ConvertSliceValue(slice.Step);
                negativeStep = (step < 0);
            }

            // Find the start
            if (slice.Start == null)
            {
                start = (npy_intp)(negativeStep ? -1 : 0);
            }
            else
            {
                start = ConvertSliceValue(slice.Start);
            }


            // Find the stop
            if (slice.Stop == null)
            {
                hasStop = false;
                stop = 0;
            }
            else
            {
                hasStop = true;
                stop = ConvertSliceValue(slice.Stop);
            }

            // Write the type
            int offset = num_indexes * sizeof(npy_intp);
            if (!hasStop)
            {
                indexes[num_indexes].type = NpyIndexType.NPY_INDEX_SLICE_NOSTOP;

                NpyIndexSliceNoStop IndexSliceNoStop = new NpyIndexSliceNoStop()
                {
                    start = start,
                    step = step,
                };

                indexes[num_indexes].slice_nostop = IndexSliceNoStop;

            }
            else
            {
                indexes[num_indexes].type = NpyIndexType.NPY_INDEX_SLICE;

                NpyIndexSlice IndexSlice = new NpyIndexSlice()
                {
                    start = start,
                    stop = stop,
                    step = step,
                };

                indexes[num_indexes].slice = IndexSlice;
            }

            ++num_indexes;
        }

        public void AddBoolArray(Object arg)
        {
            // Convert to an intp array
            ndarray arr = np.FromAny(arg, NpyCoreApi.DescrFromType(NPY_TYPES.NPY_BOOL), 0, 0, 0, null);
            // Write the type
            indexes[num_indexes].type = NpyIndexType.NPY_INDEX_BOOL_ARRAY;

            // Write the array
            indexes[num_indexes].bool_array = arr.Array;

            ++num_indexes;
        }

        public void AddIntpArray(Object arg)
        {
            // Convert to an intp array
            ndarray arr = np.FromAny(arg, NpyCoreApi.DescrFromType(NPY_TYPES.NPY_INTP), 0, 0, NPYARRAYFLAGS.NPY_FORCECAST, null);
            // Write the type
            indexes[num_indexes].type = NpyIndexType.NPY_INDEX_INTP_ARRAY;

            // Write the array
            indexes[num_indexes].intp_array = arr.Array;

            ++num_indexes;
        }

        public void AddNewAxis()
        {
            indexes[num_indexes].type = NpyIndexType.NPY_INDEX_NEWAXIS;
            ++num_indexes;
            ++num_newindexes;
        }

        public void AddEllipsis()
        {
            indexes[num_indexes].type = NpyIndexType.NPY_INDEX_ELLIPSIS;
            ++num_indexes;
        }

        internal NpyIndexType IndexType(int n)
        {
            return indexes[n].type;
        }

        public npy_intp GetIntP(int i)
        {
            return indexes[i].intp;
        }

        public bool GetBool(int i)
        {
            return indexes[i].boolean;
        }

        public bool IsAdvancedIndexing = false;
        private int num_indexes;
        private int num_newindexes = 0;
        private NpyIndex[]indexes;
        private List<string> strings;
    }
}
