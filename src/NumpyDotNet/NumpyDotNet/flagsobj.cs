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
using System.Runtime.InteropServices;
using NumpyLib;

namespace NumpyDotNet
{
    public class flagsobj
    {
        internal flagsobj(ndarray arr)
        {
            if (arr == null)
            {
                flags = NPYARRAYFLAGS.NPY_CONTIGUOUS | NPYARRAYFLAGS.NPY_OWNDATA | NPYARRAYFLAGS.NPY_FORTRAN | NPYARRAYFLAGS.NPY_ALIGNED;
            }
            else
            {
                flags = arr.Array.flags;
            }
            array = arr;
        }

        private bool ChkFlags(NPYARRAYFLAGS check) {
            return (flags&check) == check;
        }

        public bool this[string name] {
            get {
                if (name != null) {
                    switch (name.Length) {
                        case 1:
                            switch (name[0]) {
                                case 'C':
                                    return contiguous;
                                case 'F':
                                    return fortran;
                                case 'W':
                                    return writeable;
                                case 'B':
                                    return behaved;
                                case 'O':
                                    return owndata;
                                case 'A':
                                    return aligned;
                                case 'U':
                                    return updateifcopy;
                            }
                            break;
                        case 2:
                            switch (name) {
                                case "CA":
                                    return carray;
                                case "FA":
                                    return farray;
                            }
                            break;
                        case 3:
                            switch (name) {
                                case "FNC":
                                    return fnc;
                            }
                            break;
                        case 5:
                            switch (name) {
                                case "FORC":
                                    return forc;
                            }
                            break;
                        case 6:
                            switch (name) {
                                case "CARRAY":
                                    return carray;
                                case "FARRAY":
                                    return farray;
                            }
                            break;
                        case 7:
                            switch (name) {
                                case "FORTRAN":
                                    return fortran;
                                case "BEHAVED":
                                    return behaved;
                                case "OWNDATA":
                                    return owndata;
                                case "ALIGNED":
                                    return aligned;
                            }
                            break;
                        case 9:
                            switch (name) {
                                case "WRITEABLE":
                                    return writeable;
                            }
                            break;
                        case 10:
                            switch (name) {
                                case "CONTIGUOUS":
                                    return contiguous;
                            }
                            break;
                        case 12:
                            switch (name) {
                                case "UPDATEIFCOPY":
                                    return updateifcopy;
                                case "C_CONTIGUOUS":
                                    return c_contiguous;
                                case "F_CONTIGUOUS":
                                    return f_contiguous;
                            }
                            break;
                    }
                }
                throw new System.Collections.Generic.KeyNotFoundException("Unknown flag");
            }
            set {
                if (name != null) {
                    if (name == "W" || name == "WRITEABLE") {
                        writeable = value;
                        return;
                    } else if (name == "A" || name == "ALIGNED") {
                        aligned = value;
                        return;
                    } else if (name == "U" || name == "UPDATEIFCOPY") {
                        updateifcopy = value;
                        return;
                    }
                }
                throw new System.Collections.Generic.KeyNotFoundException("Unknown flag");
            }
        }

        private string ValueLine(string key, bool includeNewline=true) {
            if (includeNewline) {
                return String.Format("  {0} : {1}\n", key, this[key]);
            } else {
                return String.Format("  {0} : {1}", key, this[key]);
            }
        }

   
        public override string ToString() {
            return ValueLine("C_CONTIGUOUS") +
                ValueLine("F_CONTIGUOUS") +
                ValueLine("OWNDATA") +
                ValueLine("WRITEABLE") +
                ValueLine("ALIGNED") +
                ValueLine("UPDATEIFCOPY", includeNewline:false);
        }

        // Get only flags
        public bool contiguous { get { return ChkFlags(NPYARRAYFLAGS.NPY_CONTIGUOUS); } }
        public bool c_contiguous { get { return ChkFlags(NPYARRAYFLAGS.NPY_CONTIGUOUS); } }
        public bool f_contiguous { get { return ChkFlags(NPYARRAYFLAGS.NPY_FORTRAN); } }
        public bool fortran { get { return ChkFlags(NPYARRAYFLAGS.NPY_FORTRAN); } }
        public bool owndata { get { return ChkFlags(NPYARRAYFLAGS.NPY_OWNDATA); } }
        public bool fnc { get { return f_contiguous && !c_contiguous; } }
        public bool forc { get { return f_contiguous || c_contiguous; } }
        public bool behaved { get { return ChkFlags(NPYARRAYFLAGS.NPY_BEHAVED); } }
        public bool carray { get { return ChkFlags(NPYARRAYFLAGS.NPY_CARRAY); } }
        public bool farray { get { return ChkFlags(NPYARRAYFLAGS.NPY_FARRAY) && !c_contiguous; } }

        // get/set flags
        public bool aligned
        {
            get
            {
                return ChkFlags(NPYARRAYFLAGS.NPY_ALIGNED);
            }
            set
            {
                if (array == null)
                {
                    throw new ArgumentException("Cannot set flags on array scalars");
                }
                array.setflags(null, value, null);
            }
        }

        public bool updateifcopy
        {
            get
            {
                return ChkFlags(NPYARRAYFLAGS.NPY_UPDATEIFCOPY);
            }
            set
            {
                if (array == null)
                {
                    throw new ArgumentException("Cannot set flags on array scalars");
                }
                array.setflags(null, null, value);
            }
        }

        public bool writeable
        {
            get
            {
                return ChkFlags(NPYARRAYFLAGS.NPY_WRITEABLE);
            }
            set
            {
                if (array == null)
                {
                    throw new ArgumentException("Cannot set flags on array scalars");
                }
                array.setflags(value, null, null);
            }
        }


        private NPYARRAYFLAGS flags;
        private ndarray array;
    }
}
