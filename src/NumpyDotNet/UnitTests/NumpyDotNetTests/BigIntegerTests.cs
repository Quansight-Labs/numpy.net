using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumpyLib;


namespace NumpyDotNetTests
{
    [TestClass]
    public class BigIntegerTests : TestBaseClass
    {

        [Ignore]
        [TestMethod]
        public void xxx_BigInteger_Placeholder()
        {
            System.Numerics.BigInteger c = new System.Numerics.BigInteger(1.0);
            Console.WriteLine(c.IsZero);
            Console.WriteLine(c.IsPowerOfTwo);

            c = c * 2;

            Console.WriteLine(c);
            Console.WriteLine(c);

            var d1 = Convert.ToDecimal(c);

            var d2 = Convert.ToDouble(c);

            var cc = new System.Numerics.Complex(d2, 0);


        }
    }
}
