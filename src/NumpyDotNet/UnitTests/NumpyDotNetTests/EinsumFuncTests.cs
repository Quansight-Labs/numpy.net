using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumpyDotNet;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NumpyDotNetTests
{
    [TestClass]
    public class EinsumFuncTests : TestBaseClass
    {

        [Ignore] // not implemented yet
        [TestMethod]
        public void test_einsumpath_1()
        {
            //var xx = np.einsum_path();
            return;

        }


        [Ignore] // not implemented yet
        [TestMethod]
        public void test_einsum_1()
        {
            var xx = np.einsum();
            return;
        }

        [Ignore]
        [TestMethod]
        public void test_memory_contraints()
        {
            var outer_test = build_operands("a,b,c->abc");

            var r = np.einsum_path(outer_test.subscript, outer_test.operands, optimize: ("greedy", 0));
            print(r.path);
            print(r.string_repr);
            assert_path_equal(r.path, "einsum_path", new int[] { 0, 1, 2 });

            r = np.einsum_path(outer_test.subscript, outer_test.operands, optimize: ("optimal", 0));
            print(r.path);
            print(r.string_repr);
            assert_path_equal(r.path, "einsum_path", new int[] { 0, 1, 2 });

            var long_test = build_operands("acdf,jbje,gihb,hfac");
            r = np.einsum_path(long_test.subscript, long_test.operands, optimize: ("greedy", 0));
            print(r.path);
            print(r.string_repr);
            assert_path_equal(r.path, "einsum_path", new int[] { 0, 1, 2, 3 });

            long_test = build_operands("acdf,jbje,gihb,hfac");
            r = np.einsum_path(long_test.subscript, long_test.operands, optimize: ("optimal", 0));
            print(r.path);
            print(r.string_repr);
            assert_path_equal(r.path, "einsum_path", new int[] { 0, 1, 2, 3 });


            return;
        }

        private void assert_path_equal(IEnumerable<object> path, string v1, int[] v2)
        {
          
        }

        private Dictionary<string, int> global_size_dict = new Dictionary<string, int>()
        {
            { "a", 2 }, {"b", 3}, {"c", 4}, {"d", 5}, {"e", 4}, {"f", 3}, {"g", 2}, {"h", 6}, {"i", 5}, {"j", 4}, {"k", 3 }
        };

        private (string subscript, IEnumerable<object> operands) build_operands(string str, Dictionary<string, int> size_dict = null)
        {
            if (size_dict == null)
            {
                size_dict = global_size_dict;
            }

            string subscript = string.Format("{0}", str);
            string str1 = null;

            int arrowIndex = str.IndexOf("->");
            if (arrowIndex < 0)
            {
                str1 = subscript;
            }
            else
            {
                str1 = str.Substring(0, arrowIndex);
            }

            List<object> operands = new List<object>();

            string[] terms = str1.Split(',');

            var rnd = new np.random();

            foreach (var term in terms)
            {
                List<int> dims = new List<int>();
                foreach (var t in term)
                {
                    dims.Add(size_dict[t.ToString()]);
                }
                operands.Add(rnd.rand(new shape(dims)));
            }

            return (subscript, operands);
         
        }
    }
}
