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
    internal class NpyDict
    {
        public NpyDict()
        {
            numOfBuckets = 0;
            numOfElements = 0;
            bucketArray = new Dictionary<object, object>();
        }
        internal long numOfBuckets;
        internal long numOfElements;
        public Dictionary<object,object> bucketArray;
    };

    internal class NpyDict_Iter
    {
        internal long bucket;
        internal NpyDict_KVPair element;
    };

    internal class NpyDict_KVPair
    {
        public object key;
        public object value;
        public NpyDict_KVPair next;
    };

    internal partial class numpyinternal
    {
        internal static long NpyDict_Size(NpyDict hashTable)
        {
            return hashTable.numOfElements;
        }

        internal static void NpyDict_IterInit(NpyDict_Iter iter)
        {
            iter.bucket = -1;      /* -1 because first Next() increments to 0 */
            iter.element = null;
        }

        /*--------------------------------------------------------------------------*\
         *  NAME:
         *      NpyDict_Get() - retrieves the value of a key in a HashTable
         *  DESCRIPTION:
         *      Retrieves the value of the specified key in the specified HashTable.
         *      Uses the comparison function specified by
         *      HashTableSetKeyComparisonFunction().
         *  EFFICIENCY:
         *      O(1), assuming a good hash function and element-to-bucket ratio
         *  ARGUMENTS:
         *      hashTable    - the HashTable to search
         *      key          - the key whose value is desired
         *  RETURNS:
         *      void *       - the value of the specified key, or null if the key
         *                     doesn't exist in the HashTable
        \*--------------------------------------------------------------------------*/

        internal static object NpyDict_Get(NpyDict hashTable, object key)
        {
            if (hashTable.bucketArray.ContainsKey(key))
            {
                return hashTable.bucketArray[key];
            }
            return null;
        }


        /*--------------------------------------------------------------------------*\
         *  NAME:
         *      NpyDict_IterNext() - advances to the next element and returns it
         *  DESCRIPTION:
         *      Advances to the next element and returns that key/value pair. If
         *      the end is reached false is returned and key/value are set to null.
         *      Once the iterator is at the end, calling next will continue to
         *      result in 'false' until the iterator is re-initialized.
         *
         *      Switching hash tables or modifying the hash table without re-
         *      initializing the iterator is a bad idea.
         *
         *  EFFICIENCY:
         *      O(1)
         *  ARGUMENTS:
         *      hashTable    - the HashTable to check
         *      iter         - prior iterator position
         *      &key         - out: next key
         *      &value       - out: next value
         *  RETURNS:
         *      bool         - true if a key/value pair is returned, false if the
         *                     end has been reached.
         \*--------------------------------------------------------------------------*/
        internal static bool NpyDict_IterNext(NpyDict hashTable, NpyDict_Iter iter, NpyDict_KVPair KVPair)
        {
            /* Advance to the next element in this bucket if there is one. Otherwise
               find another bucket. */
            if (null != iter.element)
            {
                iter.element = iter.element.next;
            }
            while (null == iter.element && iter.bucket < hashTable.numOfBuckets - 1)
            {
                iter.element = (NpyDict_KVPair)hashTable.bucketArray[++iter.bucket];
            }

            if (null == iter.element)
            {
                KVPair.key = null;
                KVPair.value = null;
                return false;
            }
            KVPair.key = iter.element.key;
            KVPair.value = iter.element.value;
            return true;
        }

        /*--------------------------------------------------------------------------*\
         *  NAME:
         *      NpyDict_Destroy() - destroys an existing NpyDict
         *  DESCRIPTION:
         *      Destroys an existing NpyDict.
         *  EFFICIENCY:
         *      O(n)
         *  ARGUMENTS:
         *      NpyDict    - the NpyDict to destroy
         *  RETURNS:
         *      <nothing>
         \*--------------------------------------------------------------------------*/
        internal static void NpyDict_Destroy(NpyDict hashTable)
        {
            int i;

            for (i = 0; i < hashTable.numOfBuckets; i++)
            {
                NpyDict_KVPair pair = (NpyDict_KVPair)hashTable.bucketArray[i];
                while (pair != null)
                {
                    NpyDict_KVPair nextPair = pair.next;
                    pair.key = null;
                    pair.value = null;
                    pair = nextPair;
                }
            }

            npy_free(hashTable.bucketArray);
            npy_free(hashTable);
        }

    }

}
