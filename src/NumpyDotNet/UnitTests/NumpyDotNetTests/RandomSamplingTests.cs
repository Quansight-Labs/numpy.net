﻿using System;
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
    public class RandomSamplingTests : TestBaseClass
    {
        #region Simple Random Data
        [TestMethod]
        public void test_rand_1()
        {
            ndarray arr = np.random.rand(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);
            AssertShape(arr, 2, 3, 4);

            np.random.seed(8765);
            float f = np.random.rand();
            print(f);
            Assert.AreEqual(0.03278543f, f);

            arr = np.random.rand(5000000);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual(0.99999999331246381, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual(7.223258213784334e-08, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(0.49987999522694609, avg.GetItem(0));
        }

        [TestMethod]
        public void test_randn_1()
        {
            float fr = np.random.randn();
            ndarray arr = np.random.randn(2, 3, 4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);
            AssertShape(arr, 2, 3, 4);

            np.random.seed(1234);

            fr = np.random.randn();
            print(fr);
            Assert.AreEqual(0.471435163732f, fr);

            arr = np.random.randn(5000000);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual(5.19094054905629, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual(-5.387212036872835, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(-0.00028345468151361842, avg.GetItem(0));
        }

        [TestMethod]
        public void test_randbool_1()
        {
            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.Bool);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_BOOL);
            AssertShape(arr, 4);
            print(arr);
            //AssertArray(arr, new bool[] { false, false, false, false });

            arr = np.random.randint(20, null, new shape(4, 5), dtype: np.Bool);
            AssertShape(arr, 4, 5);
            print(arr);

            arr = np.random.randint(20, 21, new shape(2, 3), dtype: np.Bool);
            AssertShape(arr, 2, 3);
            print(arr);
            //AssertArray(arr, new SByte[,] { { 20, 20, 20 }, { 20, 20, 20 } });

            arr = np.random.randint(-2, 3, new shape(5000000), dtype: np.Bool);
            print(np.amax(arr));
            print(np.amin(arr));
            print(np.average(arr));
        }

        [TestMethod]
        public void test_randint8_1()
        {
            np.random.seed(9292);

            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.Int8);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_BYTE);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new SByte[] { 2, 2, 2, 2 });

            arr = np.random.randint(2, 8, new shape(5000000), dtype: np.Int8);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_BYTE);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((sbyte)7, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((sbyte)2, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(4.4994078, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new sbyte[] { 3, 6, 7, 5, 5, 5, 7, 7, 4, 2 });


            arr = np.random.randint(-2, 3, new shape(5000000), dtype: np.Int8);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_BYTE);

            amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((sbyte)2, amax.GetItem(0));

            amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((sbyte)(-2), amin.GetItem(0));

            avg = np.average(arr);
            print(avg);
            Assert.AreEqual(-0.0003146, avg.GetItem(0));

            first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new sbyte[] { 0, 1, -1, -2, -2, -1, 2, -1, -2, -2 });

        }



        [TestMethod]
        public void test_randuint8_1()
        {
            np.random.seed(1313);

            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.UInt8);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UBYTE);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new Byte[] { 2, 2, 2, 2 });

            arr = np.random.randint(2, 128, new shape(5000000), dtype: np.UInt8);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UBYTE);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((byte)127, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((byte)2, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(64.5080776, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new byte[] { 114, 37, 22, 53, 90, 27, 34, 98, 123, 114 });

        }

        [TestMethod]
        public void test_randint16_1()
        {
            np.random.seed(8381);

            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.Int16);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT16);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new Int16[] { 2, 2, 2, 2 });

            arr = np.random.randint(2, 2478, new shape(5000000), dtype: np.Int16);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT16);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((Int16)2477, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((Int16)2, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(1239.6188784, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new Int16[] { 1854, 1641, 1219, 1190, 1714, 644, 441, 484, 579, 393 });


            arr = np.random.randint(-2067, 3000, new shape(5000000), dtype: np.Int16);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT16);

            amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((Int16)2999, amax.GetItem(0));

            amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((Int16)(-2067), amin.GetItem(0));

            avg = np.average(arr);
            print(avg);
            Assert.AreEqual(466.3735356, avg.GetItem(0));

            first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new Int16[] {-1761, -1165, 2645, -1210, 1741, -1692, -1042, -354, 2637, -706 });
        }


        [TestMethod]
        public void test_randuint16_1()
        {
            np.random.seed(5555);

            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.UInt16);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UINT16);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new UInt16[] { 2, 2, 2, 2 });

            arr = np.random.randint(23, 12801, new shape(5000000), dtype: np.UInt16);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UINT16);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((UInt16)12800, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((UInt16)23, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(6411.5873838, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new UInt16[] { 3781, 11097, 4916, 2759, 12505, 827, 12017, 3982, 7669, 11128 });
        }

        [TestMethod]
        public void test_randint_1()
        {
            np.random.seed(701);

            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.Int32);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT32);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new Int32[] { 2, 2, 2, 2 });

            arr = np.random.randint(9, 128000, new shape(5000000), dtype: np.Int32);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT32);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((Int32)127999, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((Int32)9, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(64017.8102496, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new Int32[] { 116207, 47965, 90333, 86510, 49239, 115311, 100417, 99653, 14878, 92386 });


            arr = np.random.randint(-20000, 300000, new shape(5000000), dtype: np.Int32);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT32);

            amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((Int32)299999, amax.GetItem(0));

            amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((Int32)(-20000), amin.GetItem(0));

            avg = np.average(arr);
            print(avg);
            Assert.AreEqual(139952.5096864, avg.GetItem(0));

            first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new Int32[] { 211742, 98578, 96618, 50714, 285136, 270233, 186899, 278442, 215280, 268431 });
        }


        [TestMethod]
        public void test_randuint_1()
        {
            np.random.seed(8357);

            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.UInt32);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UINT32);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new UInt32[] { 2, 2, 2, 2 });

            arr = np.random.randint(29, 13000, new shape(5000000), dtype: np.UInt32);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UINT32);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((UInt32)12999, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((UInt32)29, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(6512.7336774, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new UInt32[] { 6511, 10414, 12909, 5962, 353, 8634, 8330, 1452, 5009, 5444 });
        }

        [TestMethod]
        public void test_randint64_1()
        {
            np.random.seed(10987);

            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.Int64);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT64);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new Int64[] { 2, 2, 2, 2 });

            arr = np.random.randint(20, 9999999, new shape(5000000), dtype: np.Int64);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT64);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((Int64)9999995, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((Int64)24, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(5000256.718873, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new Int64[] { 680213, 6755395, 2873894, 6180772, 6542022, 4878058, 9894982, 6332894, 1846901, 8550855 });


            arr = np.random.randint(-9999999, 9999999, new shape(5000000), dtype: np.Int64);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT64);

            amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((Int64)9999994, amax.GetItem(0));

            amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((Int64)(-9999994), amin.GetItem(0));

            avg = np.average(arr);
            print(avg);
            Assert.AreEqual(-3134.7668014, avg.GetItem(0));

            first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new Int64[] { 8539427, -3969151, -3294108, 7156061, 6272194, 9698115, 1323314, -4364865, -124057, -6816292 });
        }


        [TestMethod]
        public void test_randuint64_1()
        {
            np.random.seed(1990);

            ndarray arr = np.random.randint(2, 3, new shape(4), dtype: np.UInt64);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UINT64);
            AssertShape(arr, 4);
            print(arr);
            AssertArray(arr, new UInt64[] { 2, 2, 2, 2 });

            arr = np.random.randint(64, 64000, new shape(5000000), dtype: np.UInt64);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UINT64);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual((UInt64)63999, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual((UInt64)64, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(32014.4591844, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new UInt64[] { 32948, 9226, 21086, 58281, 51816, 58826, 18566, 13253, 28502, 39394 });
        }


        [TestMethod]
        public void test_rand_bytes_1()
        {
            np.random.seed(6432);

            byte br = np.random.getbyte();

            var bytes = np.random.bytes(24);
            Assert.AreEqual(bytes.Length, 24);

            br = np.random.getbyte();
            var arr = np.array(np.random.bytes(24)).reshape(2,3,4);
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_UBYTE);
            AssertShape(arr, 2, 3, 4);

            print(br);
            Assert.AreEqual(193, br);

            print(arr);

            var ExpectedData = new byte[,,]
                { { { 18, 59, 87, 188 },
                    { 116, 37, 28, 134 },
                    { 204, 39, 5, 212 } },
                  { { 147, 165, 1, 136 },
                    { 198, 108, 203, 151 },
                    { 39, 187, 144, 209 } } };

            AssertArray(arr, ExpectedData);
        }
        #endregion

        #region shuffle/permutation

        [TestMethod]
        public void test_rand_shuffle_1()
        {
            np.random.seed(1964);
            var arr = np.arange(10);
            np.random.shuffle(arr);
            print(arr);
            AssertArray(arr, new Int32[] { 2, 9, 3, 6, 1, 7, 5, 0, 4, 8 });

            arr = np.arange(10).reshape((-1,1));
            print(arr);

            np.random.shuffle(arr);
            print(arr);
            AssertArray(arr, new Int32[,] { { 0 }, { 3 }, { 7 }, { 8 }, { 5 }, { 9 }, { 4 }, { 6 }, { 1 }, { 2 } });
        }

        [TestMethod]
        public void test_rand_permutation_1()
        {
            np.random.seed(1963);

            var arr = np.random.permutation(10);
            print(arr);
            AssertArray(arr, new Int32[] { 6, 7, 4, 5, 1, 2, 9, 0, 8, 3 });

            arr = np.random.permutation(np.arange(5));
            print(arr);
            AssertArray(arr, new Int32[] { 2, 3, 1, 0, 4 });

        }

        #endregion

        [TestMethod]
        public void test_rand_beta_1()
        {
            np.random.seed(5566);

            var a = np.arange(1,11, dtype: np.Float64);
            var b = np.arange(1,11, dtype: np.Float64);

            ndarray arr = np.random.beta(b, b, new shape(10));
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_DOUBLE);
            print(arr);

            var ExpectedData = new double[]
                { 0.890517356256563, 0.532484155787344, 0.511396509650771, 0.862418837558698, 0.492458004172681,
                  0.504662615187708, 0.621477223753938, 0.41702032419038, 0.492984829781569, 0.47847289036676 };

            AssertArray(arr, ExpectedData);
        }

        [TestMethod]
        public void test_rand_binomial_1()
        {
            np.random.seed(123);

            ndarray arr = np.random.binomial(9, 0.1, new shape(20));
            AssertArray(arr, new Int64[] { 1, 0, 0, 1, 1, 1, 3, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1 });
            Assert.AreEqual(arr.TypeNum, NPY_TYPES.NPY_INT64);

            var s = np.sum(arr == 0);
            Assert.AreEqual(6, s.GetItem(0));


            arr = np.random.binomial(9, 0.1, new shape(20000));
            s = np.sum(arr == 0);
            Assert.AreEqual(7711, s.GetItem(0));
            print(s);
        }

        [TestMethod]
        public void test_rand_chisquare_1()
        {
            np.random.seed(904);

            ndarray arr = np.random.chisquare(2, new shape(40));

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual(5.375544801685989, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual(0.08589992390559097, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(1.7501237764369322, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new double[] 
            { 0.449839203939145, 1.92402228590093, 3.34447894813104, 5.34660224752626, 1.15307500835178,
              3.7142340997291, 0.137140658434128, 1.69505253573874, 1.5675310912308, 3.1550000636764 });


            arr = np.random.chisquare(np.arange(1, (25 * 25) + 1), new shape(25 * 25));

            amax = np.amax(arr);
            print(amax);
            Assert.AreEqual(686.8423283498724, amax.GetItem(0));

            amin = np.amin(arr);
            print(amin);
            Assert.AreEqual(0.5907891154976891, amin.GetItem(0));

            avg = np.average(arr);
            print(avg);
            Assert.AreEqual(313.32691732965679, avg.GetItem(0));

            first10 = arr["0:10:1"] as ndarray;
            print(first10);
            AssertArray(first10, new double[]
            { 0.590789115497689, 6.66144890165653, 4.3036608098316, 3.81142549081703, 3.43184119361682,
              7.72738961829532, 8.62984666080682, 7.93816304470877, 4.58662123588299, 10.139304988442 });

        }

        [TestMethod]
        public void test_rand_dirichlet_1()
        {
            np.random.seed(904);

            var arr = np.random.dirichlet(new int[] { 2, 20 }, 40);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual(0.9840840815128786, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual(0.01591591848712132, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(0.5, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);

            var ExpectedData = new double[,]
                { { 0.139631231507379, 0.860368768492621 },
                  { 0.164792458731906, 0.835207541268094 },
                  { 0.171803553905849, 0.828196446094151 },
                  { 0.0644690936340902, 0.93553090636591 },
                  { 0.0159159184871213, 0.984084081512879 },
                  { 0.0560190209751487, 0.943980979024851 },
                  { 0.042863749816706, 0.957136250183294 },
                  { 0.0629065979497427, 0.937093402050257 },
                  { 0.176381677869148, 0.823618322130852 },
                  { 0.173933715950931, 0.826066284049069 } };


            AssertArray(first10, ExpectedData);

            //////////////
            
            arr = np.random.dirichlet(new int[] { 25,1,25 }, 25*25);

            amax = np.amax(arr);
            print(amax);
            Assert.AreEqual(0.7045960873403417, amax.GetItem(0));

            amin = np.amin(arr);
            print(amin);
            Assert.AreEqual(2.6249646396377578e-05, amin.GetItem(0));

            avg = np.average(arr);
            print(avg);
            Assert.AreEqual(0.33333333333333354, avg.GetItem(0));

            first10 = arr["0:10:1"] as ndarray;
            print(first10);

            ExpectedData = new double[,]
                { { 0.610930043484893, 0.0193144481201467, 0.369755508394961 },
                  { 0.442089362038658, 0.0346318991277543, 0.523278738833588 },
                  { 0.614302268039855, 0.00320374569291326, 0.382493986267232 },
                  { 0.321145218655756, 0.0299626236169947, 0.648892157727249 },
                  { 0.39914499391642, 0.0791711737227786, 0.521683832360802 },
                  { 0.527685226612033, 0.00514173575597135, 0.467173037631995 },
                  { 0.454743888761555, 0.0548186350528715, 0.490437476185573 },
                  { 0.449442317277421, 0.0101774794978973, 0.540380203224682 },
                  { 0.464374656943362, 0.0130023049255488, 0.522623038131089 },
                  { 0.45310574844845, 0.00503512489512577, 0.541859126656425 } };

           AssertArray(first10, ExpectedData);

        }


        [TestMethod]
        public void test_rand_exponential_1()
        {
            np.random.seed(914);

            var arr = np.random.exponential(2.0, 40);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual(9.197509067464914, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual(0.0029207069544253516, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(2.2703450760147272, avg.GetItem(0));

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);

            var ExpectedData = new double[]
             { 0.747521091531369, 2.07455044402581, 0.266251619077963, 0.853502719631811, 6.17373567708619,
               9.19750906746491, 0.716019724979618, 0.00292070695442535, 1.07083473949392, 2.71744206583613 };

            AssertArray(first10, ExpectedData);

             //////////////

            arr = np.random.exponential(new double[] { 1.75, 2.25, 3.5, 4.1 }, 4);

            amax = np.amax(arr);
            print(amax);
            Assert.AreEqual(6.4389292872403, amax.GetItem(0));

            amin = np.amin(arr);
            print(amin);
            Assert.AreEqual(0.8405196701524901, amin.GetItem(0));

            avg = np.average(arr);
            print(avg);
            Assert.AreEqual(2.3954160266720965, avg.GetItem(0));

            first10 = arr["0:10:1"] as ndarray;
            print(first10);

            ExpectedData = new double[] { 0.84051967015249, 1.41367329358207, 0.888541855713523, 6.4389292872403 };

            AssertArray(first10, ExpectedData);

            //////////////

            arr = np.random.exponential(1.75, 200000);

            amax = np.amax(arr);
            print(amax);
            Assert.AreEqual(21.318986653331216, amax.GetItem(0));

            amin = np.amin(arr);
            print(amin);
            Assert.AreEqual(3.55593171092486e-06, amin.GetItem(0));

            avg = np.average(arr);
            print(avg);
            Assert.AreEqual(1.7531454665721702, avg.GetItem(0));

            first10 = arr["0:10:1"] as ndarray;
            print(first10);

            ExpectedData = new double[]
            { 1.36762201783298, 2.8658070450868, 0.624351703816035, 0.968550532587695, 4.27788701329609,
              0.366187067053844, 1.8701491040941, 0.322938134269804, 1.77904860176968, 0.311565039308104 };
  
            AssertArray(first10, ExpectedData);

        }

        [TestMethod]
        public void test_rand_uniform_1()
        {
            np.random.seed(5461);

            ndarray arr = np.random.uniform(-1, 1, new shape(5000000));

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual(0.9999998733604805, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual(-0.999999696666168, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(6.643562403733275E-05, avg.GetItem(0));
                            

            var first10 = arr["0:10:1"] as ndarray;
            print(first10);

            AssertArray(first10, new double[] { -0.0861951950449646, -0.916354322633752, 0.979020117468637,
                -0.196568936650096, -0.875996989224622, 0.106127858765562, 0.143469747306076, 0.375363669127145,
                -0.590520508804662, -0.18455495968856 });
        }


        [TestMethod]
        public void test_rand_standard_normal_1()
        {
            np.random.seed(8877);
            ndarray arr = np.random.standard_normal(5000000);

            var amax = np.amax(arr);
            print(amax);
            Assert.AreEqual(5.1189159770119135, amax.GetItem(0));

            var amin = np.amin(arr);
            print(amin);
            Assert.AreEqual(-5.151481557636637, amin.GetItem(0));

            var avg = np.average(arr);
            print(avg);
            Assert.AreEqual(-0.00054225209998402292, avg.GetItem(0));


            var first10 = arr["0:10:1"] as ndarray;
            print(first10);

            AssertArray(first10, new double[] { 0.345140531263051, 0.38484191742645, 1.08309197380133, 0.586824429286654,
                -1.10076748173233, 1.2922798767235, -0.604010755236405, -0.191509685425675, 0.539265713947259, 2.01982669933162 });
        }

    }
}
