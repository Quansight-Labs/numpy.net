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
    public class ComplexNumbersTests : TestBaseClass
    {
        [Ignore]
        [TestMethod]
        public void xxx_ComplexNumbers_Placeholder()
        {
            System.Numerics.Complex c = new System.Numerics.Complex(1.2, 2.0);
            Console.WriteLine(c.Real);
            Console.WriteLine(c.Imaginary);

            c = c * 2;

            Console.WriteLine(c.Real);
            Console.WriteLine(c.Imaginary);

            var d1 = Convert.ToDecimal(c);

            var d2 = Convert.ToDouble(c);

            var cc = new System.Numerics.Complex(d2, 0);
        }

        #if NOT_PLANNING_TODO
        [Ignore]
        [TestMethod]
        public void xxx_test_angle_1()
        {
        }

        [Ignore]
        [TestMethod]
        public void xxx_test_real_1()
        {
        }

        [Ignore]
        [TestMethod]
        public void xxx_test_image_1()
        {
        }
        [Ignore]
        [TestMethod]
        public void xxx_test_conj_1()
        {
        }
        [Ignore]
        [TestMethod]
        public void test_real_if_close_1()
        {

        }
        #endif

    }
}
