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
using System.Threading.Tasks;

namespace NumpyLib
{
    public class NpyObject_HEAD
    {
        public NpyObject_HEAD()
        {
            nob_type = new NpyTypeObject();
            nob_magic_number = npy_defs.NPY_VALID_MAGIC;
        }
        internal UInt32 nob_refcnt;
        internal NpyTypeObject nob_type;
        internal object nob_interface;
        internal UInt32 nob_magic_number; /* Initialized to NPY_VALID_MAGIC initialization and NPY_INVALID_MAGIC on dealloc */
    }

    delegate void npy_destructor(object o1);
    delegate int npy_wrapper_construct(object o1, ref object o2);

    internal class NpyTypeObject
    {
        internal npy_destructor ntp_dealloc = null;
        internal npy_wrapper_construct ntp_interface_alloc = null;
    }

 
    internal partial class numpyinternal
    {

        /* Returns the interface pointer for the object.  If the interface pointer is null and the interface allocator
            function is defined, the interface is created and that instance is returned.  This allows types such as
            iterators that typically don't need a wrapper to skip that step until needed. */

        internal static object Npy_INTERFACE(NpyObject_HEAD m1)
        {
            if (null != m1.nob_interface)
                return m1.nob_interface;

            if (null != m1.nob_type.ntp_interface_alloc)
            {
                m1.nob_type.ntp_interface_alloc(m1, ref m1.nob_interface);
                return m1.nob_interface;
            }

            return null;
        }
        internal static object Npy_INTERFACE(NpyArray m1)
        {
            if (null != m1.nob_interface)
                return m1.nob_interface;

            if (null != m1.nob_type.ntp_interface_alloc)
            {
                m1.nob_type.ntp_interface_alloc(m1, ref m1.nob_interface);
                return m1.nob_interface;
            }

            return null;
        }
        internal static object Npy_INTERFACE(NpyArray_Descr m1)
        {
            if (null != m1.nob_interface)
                return m1.nob_interface;

            if (null != m1.nob_type.ntp_interface_alloc)
            {
                m1.nob_type.ntp_interface_alloc(m1, ref m1.nob_interface);
                return m1.nob_interface;
            }

            return null;
        }


        internal static void NpyObject_Init(NpyObject_HEAD a, NpyTypeObject t)
        {
            a.nob_refcnt = 1;
            a.nob_type = t;
            a.nob_interface = null;
            a.nob_magic_number = npy_defs.NPY_VALID_MAGIC;
        }


        internal static NpyTypeObject NpyArray_Type = new NpyTypeObject()
        {
            ntp_dealloc = null,
            ntp_interface_alloc = null,
        };
        internal static NpyTypeObject NpyArrayMapIter_Type = new NpyTypeObject()
        {
            ntp_dealloc = arraymapiter_dealloc,
            ntp_interface_alloc = null,
        };

    }

}
