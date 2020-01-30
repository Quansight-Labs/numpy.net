using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumpyLib;
using System.Collections.Concurrent;
using System.Threading.Tasks;

namespace NumpyDotNetTests
{
    [TestClass]
    public class MultithreadedTests : TestBaseClass
    {
        class MethodInfoData
        {
            public System.Reflection.MethodInfo MethodInfo;
            public TestBaseClass baseClass;
        }

        [TestMethod]
        public void MultiThreaded_ExecuteAllUnitTests()
        {
            var Methods = GetArrayOfUnitTests();

            Random rnd = new Random();
            Methods = Methods.OrderBy(x => rnd.Next()).ToArray();

            var exceptions = new ConcurrentQueue<Exception>();

            Parallel.For(0, Methods.Count(), index =>
            {
                var m = Methods[index];

                if (m.MethodInfo.Name.StartsWith("test"))
                {
                    try
                    {
                        m.MethodInfo.Invoke(m.baseClass, null);
                    }
                    catch (Exception ex)
                    {
                        exceptions.Enqueue(ex);
                    }
                }
            });

            if (exceptions.Count > 0)
            {
                Assert.Fail(exceptions.ElementAt(0).Message);
            }

        }

        [Ignore]
        [TestMethod]
        public void MultiThreaded_ExecuteAllUnitTests_10X()
        {

            for (int i = 0; i < 10; i++)
            {
                MultiThreaded_ExecuteAllUnitTests();
            }
  
        }

        private MethodInfoData[] GetArrayOfUnitTests()
        {
            List<MethodInfoData> MethodInfo = new List<MethodInfoData>();

            MethodInfo.AddRange(GetArrayOfUnitTests<ArrayConversionsTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<ArrayCreationTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<BigIntegerTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<ComplexNumbersTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<DecimalNumbersTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<FromNumericTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<IndexTricksTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<IteratorTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<MathematicalFunctionsTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<NANFunctionsTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<NumericOperationsTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<NumericTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<ObjectOperationsTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<ShapeBaseTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<StatisticsTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<StrideTricksTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<TwoDimBaseTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<UFUNCTests>());
            MethodInfo.AddRange(GetArrayOfUnitTests<WindowFunctions>());


            return MethodInfo.ToArray();
        }

        private MethodInfoData[] GetArrayOfUnitTests<T>() where T: TestBaseClass, new()
        {
            List<MethodInfoData> MethodInfo = new List<MethodInfoData>();

            T UnitTests = new T();
            var Methods = UnitTests.GetType().GetMethods();
            foreach (var m in Methods)
            {
                if (m.CustomAttributes.FirstOrDefault(t=>t.AttributeType.Name.Contains("Ignore")) == null)
                {
                    MethodInfo.Add(new MethodInfoData() { MethodInfo = m, baseClass = UnitTests });
                }
            }

            return MethodInfo.ToArray();
        }
    }
}
