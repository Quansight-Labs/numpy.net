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
using System.Reflection;
using System.Reflection.Emit;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Tests
{
    public class FastCopy
    {
        static readonly ICopier _copier;

        protected static AssemblyName _asmName = new AssemblyName() { Name = "FastCopier" };
        protected static ModuleBuilder _modBuilder;
        protected static AssemblyBuilder _asmBuilder;

        static FastCopy()
        {
            _asmBuilder = Thread.GetDomain().DefineDynamicAssembly(_asmName, AssemblyBuilderAccess.RunAndSave);
            _modBuilder = _asmBuilder.DefineDynamicModule(_asmName.Name, _asmName.Name + ".dll", true);

            var typeBuilder = _modBuilder.DefineType("FastCopier",
                       TypeAttributes.Public
                       | TypeAttributes.AutoClass
                       | TypeAttributes.AnsiClass
                       | TypeAttributes.Class
                       | TypeAttributes.Serializable
                       | TypeAttributes.BeforeFieldInit);
            typeBuilder.AddInterfaceImplementation(typeof(ICopier));
            var copyMethod = typeBuilder.DefineMethod("Copy",
                MethodAttributes.Public | MethodAttributes.HideBySig | MethodAttributes.Virtual,
                typeof(void),
                new Type[] { typeof(byte[]), typeof(byte[]), typeof(int), typeof(uint) });
            var code = copyMethod.GetILGenerator();

            code.Emit(OpCodes.Ldarg_2);
            code.Emit(OpCodes.Ldc_I4_0);
            code.Emit(OpCodes.Ldelema, typeof(byte));
            code.Emit(OpCodes.Ldarg_1);
            code.Emit(OpCodes.Ldarg_3);
            code.Emit(OpCodes.Ldelema, typeof(byte));
            code.Emit(OpCodes.Ldarg, 4);
            code.Emit(OpCodes.Cpblk);
            code.Emit(OpCodes.Ret);

            typeBuilder.DefineMethodOverride(copyMethod, typeof(ICopier).GetMethod("Copy"));

            var copierType = typeBuilder.CreateType();
            _copier = (ICopier)Activator.CreateInstance(copierType);

        }

        public static void Copy(byte[] source, byte[] dest, int offset, uint count)
        {
            _copier.Copy(source, dest, offset, count);
        }
    }

    public interface ICopier
    {
        void Copy(byte[] source, byte[] dest, int offset, uint count);
    }
}
