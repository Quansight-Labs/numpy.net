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
    public class DateTimeSupportTests : TestBaseClass
    {

        [TestMethod]
        public void test_DateTime_Basic_1()
        {
            DateTime Now = DateTime.Now;
            DateTime NowPlusOneHour = Now.AddHours(1);

            var now = np.array(new object[] { Now, Now, Now });
            var nowplus = np.array(new object[] { NowPlusOneHour, NowPlusOneHour, NowPlusOneHour });

            var timediff = nowplus - now;

            foreach (var td in timediff)
            {
                TimeSpan ts = (TimeSpan)td;
                print(ts);
            }

            print(timediff);

            print("************");


        }


        [TestMethod]
        public void test_DateTime_Basic_2()
        {
            DateTime Now = DateTime.Now;
            DateTime NowPlusOneHour = Now.AddHours(1);

            var now = np.array(new object[] { Now, Now, Now });
            now.Name = "NOW";
            print(now);

            var tomorrow = now + new TimeSpan(24, 0, 0);
            tomorrow.Name = "TOMORROW";

            print(tomorrow);
     

        }

    }
}
