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
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif
namespace NumpyLib
{

    public class NpyArray_Descr_serializable
    {
        public char kind;      /* kind for this type */
        public byte type;              /* unique-character representing this type */
        public char byteorder;         /*
                                        * '>' (big), '<' (little), '|'
                                        * (not-applicable), or '=' (native).
                                        */
        public int flags;  /* flag describing data type */
        public NPY_TYPES type_num;      /* number representing this type */


        public int elsize;     /* element size for this type */

        public int eldivshift;
        public int alignment;          /* alignment needed for this type */
                                       //public NpyArray_ArrayDescr subarray;          /*
                                       //                        * Non-null if this type is
                                       //                        * is an array (C-contiguous)
                                       //                        * of some other type
                                       //                        */

        public NpyArray_Descr_serializable subarray;
        public int subarray_shape_num_dims;  
        public npy_intp[] subarray_shape_dims;      


        public List<string> names;      /* Array of char *, null indicates end of array.
                                * char* lifetime is exactly lifetime of array
                                * itself. */
    }


    internal  class NpyArray_Descr : NpyObject_HEAD
    {
   
        public NpyArray_Descr(NPY_TYPES type_num)
        {
            f = numpyinternal.GetArrFuncs(type_num);
            byteorder = numpyinternal.NPY_NATIVE;
            this.type_num = type_num;

            this.elsize = DefaultArrayHandlers.GetArrayHandler(type_num).ItemSize;

            this.f.nonzero = DefaultArrayHandlers.GetArrayHandler(type_num).NonZero;

        }

        public static NpyArray_Descr FromSerializable(NpyArray_Descr_serializable serializable)
        {
            NpyArray_Descr descr = new NpyArray_Descr(serializable.type_num);
            descr.kind = (NPY_TYPECHAR)serializable.kind;
            descr.type = serializable.type;
            descr.byteorder = serializable.byteorder;
            descr.flags = (NpyArray_Descr_Flags)serializable.flags;
            descr.alignment = serializable.alignment;
            descr.names = serializable.names;

            return descr;
        }

        public NpyArray_Descr_serializable ToSerializable()
        {
            var serializable = new NpyArray_Descr_serializable()
            {
                kind = (char)this.kind,
                type = this.type,
                byteorder = this.byteorder,
                flags = (int)this.flags,
                type_num = this.type_num,
                elsize = this.elsize,
                alignment = this.alignment,
                names = this.names,
            };

            if (this.subarray != null)
            {
                serializable.subarray = this.subarray._base.ToSerializable();
                serializable.subarray_shape_dims = this.subarray.shape_dims;
                serializable.subarray_shape_num_dims = this.subarray.shape_num_dims;
            }
            else
            {
                serializable.subarray = null;
                serializable.subarray_shape_dims = null;
                serializable.subarray_shape_num_dims = 0;
            }

            return serializable;
        }

        public NPY_TYPECHAR kind;              /* kind for this type */
        public byte type;              /* unique-character representing this type */
        public char byteorder;         /*
                                 * '>' (big), '<' (little), '|'
                                 * (not-applicable), or '=' (native).
                                */
        internal byte unused;
        public NpyArray_Descr_Flags flags;              /* flag describing data type */
        public NPY_TYPES type_num;           /* number representing this type */

        private int _elsize;
        public int elsize      /* element size for this type */
        {
            get
            {
                return _elsize;
            }

            set
            {
                _elsize = value;
                eldivshift = numpyinternal.GetDivSize(value);
            }
        }
        public int eldivshift;
        public int alignment;          /* alignment needed for this type */
        public NpyArray_ArrayDescr
            subarray;          /*
                                * Non-null if this type is
                                * is an array (C-contiguous)
                                * of some other type
                                */
        public NpyDict
            fields;            /* The fields dictionary for this type
                                * For statically defined descr this
                                * is always null.
                                */

        public List<string> names;      /* Array of char *, null indicates end of array.
                                * char* lifetime is exactly lifetime of array
                                * itself. */

        public NpyArray_ArrFuncs f; /*
                              * a table of functions specific for each
                              * basic data descriptor
                              */


    }
    [Flags]
    internal enum NpyArray_Descr_Flags: int
    {
   

    }

    internal partial class numpyinternal
    {
        internal static NpyTypeObject NpyArrayDescr_Type = new NpyTypeObject()
        {
            ntp_dealloc = NpyArray_DescrDestroy,
            ntp_interface_alloc = null,
        };

        internal static NpyArray_Descr NpyArray_DescrNew(NpyArray_Descr origDescr)
        {
            NpyArray_Descr newDescr;

            Debug.Assert(Validate(origDescr));

            newDescr = new NpyArray_Descr(origDescr.type_num);
            if (newDescr == null)
            {
                return null;
            }
            NpyObject_Init(newDescr, NpyArrayDescr_Type);

            newDescr.kind = origDescr.kind;
            newDescr.type = origDescr.type;
            newDescr.byteorder = origDescr.byteorder;
            newDescr.unused = origDescr.unused;
            newDescr.flags = origDescr.flags;
            newDescr.type_num = origDescr.type_num;
            newDescr.elsize = origDescr.elsize;
            newDescr.alignment = origDescr.alignment;

            Debug.Assert((null == newDescr.fields && null == newDescr.names) ||
                   (null != newDescr.fields && null != newDescr.names));

            if (newDescr.fields != null)
            {
                newDescr.names = NpyArray_DescrNamesCopy(newDescr.names);
                newDescr.fields = NpyDict_Copy(origDescr.fields);
            }
            if (newDescr.subarray != null)
            {
                newDescr.subarray = NpyArray_DupSubarray(origDescr.subarray);
            }
 
            /* Allocate the interface wrapper object. */
            if (false == NpyInterface_DescrNewFromWrapper(Npy_INTERFACE(origDescr), newDescr, ref newDescr.nob_interface))
            {
                newDescr.nob_interface = null;
                Npy_DECREF(newDescr);
                return null;
            }

            /* Note on reference counts: At this point if there is an inteface object,
               it's refcnt should be == 1 because the refcnt on the core object == 1.
               That is, the core object is holding a single reference to the interface
               object. */
            return newDescr;
        }


        internal static NpyArray_Descr NpyArray_DescrNewSubarray(NpyArray_Descr baseDescr, int ndim, npy_intp[] dims)
        {
            NpyArray_Descr result = NpyArray_DescrNewFromType(NPY_TYPES.NPY_OBJECT);
            if (result == null)
            {
                return result;
            }
            result.elsize = baseDescr.elsize;
            result.elsize *= (int)NpyArray_MultiplyList(dims, ndim);
            NpyArray_DescrReplaceSubarray(result, baseDescr, ndim, dims);
            result.flags = baseDescr.flags;
            /* I'm not sure why or if the next call is needed, but it is in the CPython code. */
            NpyArray_DescrDeallocNamesAndFields(result);

            return result;
        }


        internal static void NpyArray_DescrReplaceSubarray(NpyArray_Descr descr, NpyArray_Descr baseDescr, int ndim, npy_intp []dims)
        {
            Debug.Assert(Validate(descr));
            Debug.Assert(Validate(baseDescr));

            if (null != descr.subarray)
            {
                NpyArray_DestroySubarray(descr.subarray);
            }

            descr.subarray = new NpyArray_ArrayDescr();
            descr.subarray._base = baseDescr;
            Npy_INCREF(baseDescr);
            descr.subarray.shape_num_dims = ndim;
            descr.subarray.shape_dims = new npy_intp[ndim];
            copydims(descr.subarray.shape_dims, dims, ndim);
        }

        /*
         * self cannot be null
         * Destroys the given descriptor and deallocates the memory for it.
         */
        internal static void NpyArray_DescrDestroy(object o1)
        {
            NpyArray_Descr self = (NpyArray_Descr)o1;

            Debug.Assert(Validate(self));

            NpyArray_DescrDeallocNamesAndFields(self);

            if (self.subarray != null)
            {
                NpyArray_DestroySubarray(self.subarray);
                self.subarray = null;
            }
  
            self.nob_magic_number = npy_defs.NPY_INVALID_MAGIC;

            npy_free(self);
        }

        /*
         * self cannot be null
         */
        internal static void NpyArray_DescrDeallocNamesAndFields(NpyArray_Descr self)
        {
            int i;

            if (null != self.names)
            {
                for (i = 0; null != self.names[i]; i++)
                {
                    if (self.names[i] != null)
                    {
                        npy_free(self.names[i]);
                    }
                    self.names[i] = null;
                }
                npy_free(self.names);
                self.names = null;
            }

            if (null != self.fields)
            {
                self.fields = null;
            }
        }

        private static NpyDict NpyDict_Copy(NpyDict origFields)
        {
            NpyDict newFields = new NpyDict()
            {
                numOfBuckets = origFields.numOfBuckets,
                numOfElements = origFields.numOfElements,
                bucketArray = NpDictionaryClone(origFields.bucketArray),
            };

            return newFields;
        }

        private static Dictionary<object, object> NpDictionaryClone(Dictionary<object, object> origDict)
        {
            Dictionary<object, object> newDict = new Dictionary<object, object>();
            foreach (var kvp in origDict)
            {
                newDict.Add(kvp.Key, kvp.Value);
            }

            return newDict;
        }

        /*
         * base cannot be null
         * Allocates a new NpyDict contaning NpyArray_DescrField value object and
         * performs a deep-copy of the passed-in fields structure to populate them.
         * The descriptor field structure contains a pointer to another
         * NpyArray_Descr instance, which must be reference counted.
         */
        internal static List<string> NpyArray_DescrNamesCopy(List<string> names)
        {
            return names.ToArray().ToList();
        }

        internal static NpyArray_ArrayDescr NpyArray_DupSubarray(NpyArray_ArrayDescr src)
        {
            NpyArray_ArrayDescr dest;

            dest = new NpyArray_ArrayDescr();

            Debug.Assert((0 == src.shape_num_dims && null == src.shape_dims) ||
                   (0 < src.shape_num_dims && null != src.shape_dims));

            dest._base = src._base;
            Npy_INCREF(dest._base);

            dest.shape_num_dims = src.shape_num_dims;
            if (0 < dest.shape_num_dims)
            {
                dest.shape_dims = new npy_intp[dest.shape_num_dims];
                copydims(dest.shape_dims, src.shape_dims, dest.shape_num_dims);
            }
            else
            {
                dest.shape_dims = null;
            }
            return dest;
        }



        internal static void NpyArray_DestroySubarray(NpyArray_ArrayDescr self)
        {
            Npy_DECREF(self._base);
            if (0 < self.shape_num_dims)
            {
                NpyArray_free(self.shape_dims);
            }
            self.shape_dims = null;
            NpyArray_free(self);
        }



        private static bool NpyInterface_DescrNewFromWrapper(object v, NpyArray_Descr newDescr, ref object nob_interface)
        {
            if (_NpyArrayWrapperFuncs == null)
            {
                throw new Exception("_NpyArrayWrapperFuncs is null.  Improper initialization detected.");
            }

            return (null != _NpyArrayWrapperFuncs.descr_new_from_wrapper ? _NpyArrayWrapperFuncs.descr_new_from_wrapper(v, newDescr, ref nob_interface) : true);
        }


        internal static bool NpyDataType_FLAGCHK(NpyArray_Descr dtype, NpyArray_Descr_Flags flag)
        {
            return ((dtype.flags & flag) != 0);
        }

        internal static NpyArray_Descr NpyArray_DescrNewFromType(NPY_TYPES type_num)
        {
            NpyArray_Descr old;
            NpyArray_Descr _new;

            old = NpyArray_DescrFromType(type_num);
            _new = NpyArray_DescrNew(old);
            Npy_DECREF(old);
            return _new;
        }


        internal static NpyArray_Descr NpyArray_SmallType(NpyArray_Descr chktype, NpyArray_Descr mintype)
        {
            NpyArray_Descr outtype;
            NPY_TYPES outtype_num;
            NPY_TYPES save_num;

            Debug.Assert(Validate(chktype) && Validate(mintype));

            if (NpyArray_EquivTypes(chktype, mintype))
            {
                Npy_INCREF(mintype);
                return mintype;
            }


            if (chktype.type_num > mintype.type_num)
            {
                outtype_num = chktype.type_num;
            }
            else
            {
                if (NpyTypeNum_ISOBJECT(chktype.type_num) &&
                    NpyDataType_ISSTRING(mintype))
                {
                    return NpyArray_DescrFromType(NPY_TYPES.NPY_OBJECT);
                }
                else
                {
                    outtype_num = mintype.type_num;
                }
            }

            save_num = outtype_num;
            while (outtype_num < NPY_TYPES.NPY_NTYPES &&
                   !(NpyArray_CanCastSafely(chktype.type_num, outtype_num)
                     && NpyArray_CanCastSafely(mintype.type_num, outtype_num)))
            {
                outtype_num++;
            }
            if (outtype_num == NPY_TYPES.NPY_NTYPES)
            {
                outtype = NpyArray_DescrFromType(save_num);
            }
            else
            {
                outtype = NpyArray_DescrFromType(outtype_num);
            }
            if (NpyTypeNum_ISEXTENDED(outtype.type_num))
            {
                int testsize = outtype.elsize;
                int chksize, minsize;
                chksize = chktype.elsize;
                minsize = mintype.elsize;
 
                testsize = Math.Max(chksize, minsize);
                if (testsize != outtype.elsize)
                {
                    NpyArray_DESCR_REPLACE(ref outtype);
                    outtype.elsize = testsize;
                    NpyArray_DescrDeallocNamesAndFields(outtype);
                }
            }
            return outtype;
        }

        internal static NpyArray_Descr NpyArray_DescrFromArray(NpyArray ap, NpyArray_Descr mintype)
        {
            NpyArray_Descr chktype = null;
            NpyArray_Descr outtype = null;

            Debug.Assert(Validate(ap) && Validate(mintype));

            chktype = NpyArray_DESCR(ap);
            Npy_INCREF(chktype);
            if (mintype == null)
            {
                return chktype;
            }
            Npy_INCREF(mintype);

            outtype = NpyArray_SmallType(chktype, mintype);
            Npy_DECREF(chktype);
            Npy_DECREF(mintype);

            return outtype;
        }


        internal static void NpyArray_DescrDestroy(NpyArray_Descr self)
        {
            Debug.Assert(Validate(self));

            NpyArray_DescrDeallocNamesAndFields(self);

            if (self.subarray != null)
            {
                NpyArray_DestroySubarray(self.subarray);
                self.subarray = null;
            }
 
            self.nob_magic_number = npy_defs.NPY_INVALID_MAGIC;

            npy_free(self);
        }

        internal static NpyArray_Descr NpyArray_DescrNewByteorder(NpyArray_Descr self, char newendian)
        {
            NpyArray_Descr _new;
            char endian;

            _new = NpyArray_DescrNew(self);
            endian = _new.byteorder;
            if (endian != NPY_IGNORE)
            {
                if (newendian == NPY_SWAP)
                {
                    /* swap byteorder */
                    if (NpyArray_ISNBO(endian))
                    {
                        endian = NPY_OPPBYTE;
                    }
                    else
                    {
                        endian = NPY_NATBYTE;
                    }
                    _new.byteorder = endian;
                }
                else if (newendian != NPY_IGNORE)
                {
                    _new.byteorder = newendian;
                }
            }
            if (null != _new.names)
            {
                NpyDict_KVPair KVPair = new NpyDict_KVPair();
                NpyArray_Descr newdescr;
                NpyDict_Iter pos = new NpyDict_Iter();

                NpyDict_IterInit(pos);
                while (NpyDict_IterNext(_new.fields, pos, KVPair))
                {
                    NpyArray_DescrField value = KVPair.value as NpyArray_DescrField;
                    string key = KVPair.key as string;

                    if (null != value.title && key.CompareTo(value.title) != 0)
                    {
                        continue;
                    }
                    newdescr = NpyArray_DescrNewByteorder(value.descr, newendian);
                    if (newdescr == null)
                    {
                        Npy_DECREF(_new);
                        return null;
                    }
                    Npy_DECREF(value.descr);
                    value.descr = newdescr;
                }
            }
            if (null != _new.subarray)
            {
                NpyArray_Descr old = _new.subarray._base;
                _new.subarray._base = NpyArray_DescrNewByteorder(self.subarray._base, newendian);
                Npy_DECREF(old);
            }
            return _new;
        }

        internal static List<string> NpyArray_DescrAllocNames(int n)
        {
            List<string> Names = new List<string>();
            return Names;
        }
        internal static NpyDict NpyArray_DescrAllocFields()
        {
            return new NpyDict();
        }

        internal static int NpyArray_DescrReplaceNames(NpyArray_Descr self, List<string> nameslist)
        {
            self.names = NpyArray_DescrNamesCopy(nameslist);
            return 1;
        }

        internal static void NpyArray_DescrSetNames(NpyArray_Descr self, List<string> nameslist)
        {
            self.names = NpyArray_DescrNamesCopy(nameslist);
        }
        internal static void NpyArray_DescrSetField(NpyDict self, string key, NpyArray_Descr descr, int offset, string title)
        {
            NpyArray_DescrField field;

            field = new NpyArray_DescrField();
            field.descr = descr;
            field.offset = offset;
            field.title = title;

            if (descr.fields.bucketArray.ContainsKey(key))
            {
                descr.fields.bucketArray[key] = field;
            }
            else
            {
                descr.fields.bucketArray.Add(key, field);
                descr.fields.numOfElements++;
            }

        }
        internal static NpyDict NpyArray_DescrFieldsCopy(NpyDict fields)
        {
            return NpyDict_Copy(fields);
        }


        internal static bool npy_arraydescr_isnative(NpyArray_Descr self)
        {
            if (self.names == null)
            {
                return NpyArray_ISNBO(self);
            }
            else
            {
                NpyDict_Iter pos = new NpyDict_Iter();
                NpyDict_KVPair KVPair = new NpyDict_KVPair();
                KVPair.key = null;
                KVPair.value = null;

                NpyDict_IterInit(pos);
                while (NpyDict_IterNext(self.fields, pos, KVPair))
                {
                    string key = (string)KVPair.key;
                    NpyArray_DescrField value = (NpyArray_DescrField)KVPair.value;
                    if (null != value.title && value.title.CompareTo(key) != 0)
                    {
                        continue;
                    }
                    if (false == npy_arraydescr_isnative(value.descr))
                    {
                        return false;
                    }
                }
            }
            return true;
        }
       
    }



    internal class NpyArray_ArrFuncs
    {
        /* The next four functions *cannot* be null */

        /*
         * Functions to get and set items with standard Python types
         * -- not array scalars
         */
        public NpyArray_GetItemFunc getitem;
        public NpyArray_SetItemFunc setitem;

 
        /*
         * Function to compare items
         * Can be null
         */
        public NpyArray_CompareFunc compare;

        /*
         * Function to select largest
         * Can be null
         */
        public NpyArray_ArgFunc argmax;

        /*
        * Function to select largest
        * Can be null
        */
        public NpyArray_ArgFunc argmin;

        /*
         * Function to scan an ASCII file and
         * place a single value plus possible separator
         * Can be null
         */
        public NpyArray_ScanFunc scanfunc;

        /*
         * Function to read a single value from a string
         * and adjust the pointer; Can be null
         */
        public NpyArray_FromStrFunc fromstr;

        /*
         * Function to determine if data is zero or not
         * If null a default version is
         * used at Registration time.
         */
        public NpyArray_NonzeroFunc nonzero;

        /*
         * Used for arange.
         * Can be null.
         */
        public NpyArray_FillFunc fill;

        /*
         * Function to fill arrays with scalar values
         * Can be null
         */
        public NpyArray_FillWithScalarFunc fillwithscalar;

        /*
         * Sorting functions
         * Can be null
         */
        public NpyArray_SortFunc[]sort = new NpyArray_SortFunc[npy_defs.NPY_NSORTS];
        public NpyArray_ArgSortFunc[]argsort = new NpyArray_ArgSortFunc[npy_defs.NPY_NSORTS];

        /*
         * Array of PyArray_CastFuncsItem given cast functions to
         * user defined types. The array it terminated with PyArray_NOTYPE.
         * Can be null.
         */
        public List<NpyArray_CastFuncsItem> castfuncs = null;

        /*
         * Functions useful for generalizing
         * the casting rules.
         * Can be null;
         */
        public NpyArray_ScalarKindFunc scalarkind;
        public Dictionary<NPY_SCALARKIND, object> cancastscalarkindto;
        public List<NPY_TYPES> cancastto = null;


        /*
         * A little room to grow --- should use generic function
         * interface for most additions
         */
        object pad1;
        object pad2;
        object pad3;
        object pad4;

        /*
         * Functions to cast to all other standard types
         * Can have some null entries
         */
        public NpyArray_VectorUnaryFunc[] cast = new NpyArray_VectorUnaryFunc[(int)NPY_TYPES.NPY_NTYPES];
    }

    internal class NpyArray_ArrayDescr
    {
        public NpyArray_Descr _base;
        public int shape_num_dims;    /* shape_num_dims and shape_dims essentially
                                   implement */
        public npy_intp[] shape_dims;       /* a tuple. When shape_num_dims  >= 1
                                   shape_dims is an */
                                /* allocated array of ints; shape_dims == null iff */
                                /* shape_num_dims == 1 */
    }

    /* Used as the value of an NpyDict to record the fields in an
   NpyArray_Descr object*/
    internal class NpyArray_DescrField
    {
        public NpyArray_Descr descr;
        public int offset;
        public string title;                /* String owned/managed by each instance */
    };

    internal class NpyArray_CastFuncsItem
    {
        public NPY_TYPES totype;
        public NpyArray_VectorUnaryFunc castfunc;
    };

    internal class NpyArray_FunctionDefs
    {
        /* Get-set methods per type. */
        public NpyArray_GetItemFunc BOOL_getitem;
        public NpyArray_GetItemFunc BYTE_getitem;
        public NpyArray_GetItemFunc UBYTE_getitem;
        public NpyArray_GetItemFunc SHORT_getitem;
        public NpyArray_GetItemFunc USHORT_getitem;
        public NpyArray_GetItemFunc INT_getitem;
        public NpyArray_GetItemFunc LONG_getitem;
        public NpyArray_GetItemFunc UINT_getitem;
        public NpyArray_GetItemFunc ULONG_getitem;
        public NpyArray_GetItemFunc LONGLONG_getitem;
        public NpyArray_GetItemFunc ULONGLONG_getitem;
        public NpyArray_GetItemFunc FLOAT_getitem;
        public NpyArray_GetItemFunc DOUBLE_getitem;
        public NpyArray_GetItemFunc LONGDOUBLE_getitem;
        public NpyArray_GetItemFunc CFLOAT_getitem;
        public NpyArray_GetItemFunc CDOUBLE_getitem;
        public NpyArray_GetItemFunc CLONGDOUBLE_getitem;
        public NpyArray_GetItemFunc UNICODE_getitem;
        public NpyArray_GetItemFunc STRING_getitem;
        public NpyArray_GetItemFunc OBJECT_getitem;
        public NpyArray_GetItemFunc VOID_getitem;
        public NpyArray_GetItemFunc DATETIME_getitem;
        public NpyArray_GetItemFunc TIMEDELTA_getitem;

        public NpyArray_SetItemFunc BOOL_setitem;
        public NpyArray_SetItemFunc BYTE_setitem;
        public NpyArray_SetItemFunc UBYTE_setitem;
        public NpyArray_SetItemFunc SHORT_setitem;
        public NpyArray_SetItemFunc USHORT_setitem;
        public NpyArray_SetItemFunc INT_setitem;
        public NpyArray_SetItemFunc LONG_setitem;
        public NpyArray_SetItemFunc UINT_setitem;
        public NpyArray_SetItemFunc ULONG_setitem;
        public NpyArray_SetItemFunc LONGLONG_setitem;
        public NpyArray_SetItemFunc ULONGLONG_setitem;
        public NpyArray_SetItemFunc FLOAT_setitem;
        public NpyArray_SetItemFunc DOUBLE_setitem;
        public NpyArray_SetItemFunc LONGDOUBLE_setitem;
        public NpyArray_SetItemFunc CFLOAT_setitem;
        public NpyArray_SetItemFunc CDOUBLE_setitem;
        public NpyArray_SetItemFunc CLONGDOUBLE_setitem;
        public NpyArray_SetItemFunc UNICODE_setitem;
        public NpyArray_SetItemFunc STRING_setitem;
        public NpyArray_SetItemFunc OBJECT_setitem;
        public NpyArray_SetItemFunc VOID_setitem;
        public NpyArray_SetItemFunc DATETIME_setitem;
        public NpyArray_SetItemFunc TIMEDELTA_setitem;

        /* Object type methods. */
        public NpyArray_CopySwapNFunc OBJECT_copyswapn;
        public NpyArray_CompareFunc OBJECT_compare;
        public NpyArray_ArgFunc OBJECT_argmax;
        public NpyArray_ScanFunc OBJECT_scanfunc;
        public NpyArray_FromStrFunc OBJECT_fromstr;
        public NpyArray_NonzeroFunc OBJECT_nonzero;
        public NpyArray_FillFunc OBJECT_fill;
        public NpyArray_FillWithScalarFunc OBJECT_fillwithscalar;
        public NpyArray_ScalarKindFunc OBJECT_scalarkind;
        public NpyArray_FastClipFunc OBJECT_fastclip;
        public NpyArray_FastTakeFunc OBJECT_fasttake;

        /* Unboxing (object-to-type) */
        public NpyArray_VectorUnaryFunc[]cast_from_obj = new NpyArray_VectorUnaryFunc[(int)NPY_TYPES.NPY_NTYPES];
        /* String-to-type */
        public NpyArray_VectorUnaryFunc[] cast_from_string = new NpyArray_VectorUnaryFunc[(int)NPY_TYPES.NPY_NTYPES];
        /* Unicode-to-type */
        public NpyArray_VectorUnaryFunc[] cast_from_unicode = new NpyArray_VectorUnaryFunc[(int)NPY_TYPES.NPY_NTYPES];
        /* Void-to-type */
        public NpyArray_VectorUnaryFunc[] cast_from_void = new NpyArray_VectorUnaryFunc[(int)NPY_TYPES.NPY_NTYPES];

        /* Boxing (type-to-object) */
        public NpyArray_VectorUnaryFunc[] cast_to_obj = new NpyArray_VectorUnaryFunc[(int)NPY_TYPES.NPY_NTYPES];
        /* Type-to-string */
        public NpyArray_VectorUnaryFunc[] cast_to_string = new NpyArray_VectorUnaryFunc[(int)NPY_TYPES.NPY_NTYPES];
        /* Type-to-unicode */
        public NpyArray_VectorUnaryFunc[] cast_to_unicode = new NpyArray_VectorUnaryFunc[(int)NPY_TYPES.NPY_NTYPES];
        /* Type-to-void */
        public NpyArray_VectorUnaryFunc[] cast_to_void = new NpyArray_VectorUnaryFunc[(int)NPY_TYPES.NPY_NTYPES];

        public int sentinel;       /* Not used except to test validity of structure sizes */
    };
}
