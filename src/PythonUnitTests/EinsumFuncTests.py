import unittest
import numpy as np
from nptest import nptest
from nptest3 import nptest3

chars = 'abcdefghij'
sizes = np.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3])
global_size_dict = {}
for size, char in zip(sizes, chars):
    global_size_dict[char] = size

class EinsumFuncTests(unittest.TestCase):


    def test_einsumpath_1(self):
        self.fail("Not implemented")


    def test_einsum_1(self):
        self.fail("Not implemented")

    def build_operands(self, string, size_dict=global_size_dict):

        # Builds views based off initial operands
        operands = [string]
        terms = string.split('->')[0].split(',')
        for term in terms:
            dims = [size_dict[x] for x in term]
            operands.append(np.random.rand(*dims))

        return operands

    def assert_path_equal(self, comp, benchmark):
        # Checks if list of tuples are equivalent
        ret = (len(comp) == len(benchmark))
        assert(ret)
        for pos in range(len(comp) - 1):
            ret &= isinstance(comp[pos + 1], tuple)
            ret &= (comp[pos + 1] == benchmark[pos + 1])
        assert(ret)

    def test_memory_contraints(self):
        # Ensure memory constraints are satisfied

        outer_test = self.build_operands('a,b,c->abc')

        path, path_str = nptest3.einsum_path(*outer_test, optimize=('greedy', 0))
        print(path)
        print(path_str)
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2)])

        path, path_str = nptest3.einsum_path(*outer_test, optimize=('optimal', 0))
        print(path)
        print(path_str)
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2)])

        long_test = self.build_operands('acdf,jbje,gihb,hfac')
        path, path_str = nptest3.einsum_path(*long_test, optimize=('greedy', 0))
        print(path)
        print(path_str)
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])

        path, path_str = nptest3.einsum_path(*long_test, optimize=('optimal', 0))
        print(path)
        print(path_str)
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])

    def test_long_paths(self):
        # Long complex cases

        # Long test 1
        long_test1 = self.build_operands('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        path, path_str = nptest3.einsum_path(*long_test1, optimize='greedy')
        print(path)
        print(path_str)
        self.assert_path_equal(path, ['einsum_path',
                                      (1, 4), (2, 4), (1, 4), (1, 3), (1, 2), (0, 1)])

        path, path_str = nptest3.einsum_path(*long_test1, optimize='optimal')
        print(path)
        print(path_str)
        self.assert_path_equal(path, ['einsum_path',
                                      (3, 6), (3, 4), (2, 4), (2, 3), (0, 2), (0, 1)])

        # Long test 2
        long_test2 = self.build_operands('chd,bde,agbc,hiad,bdi,cgh,agdb')
        path, path_str = nptest3.einsum_path(*long_test2, optimize='greedy')
        print(path)
        print(path_str)
        self.assert_path_equal(path, ['einsum_path',
                                      (3, 4), (0, 3), (3, 4), (1, 3), (1, 2), (0, 1)])

        path, path_str = nptest3.einsum_path(*long_test2, optimize='optimal')
        print(path)
        print(path_str)
        self.assert_path_equal(path, ['einsum_path',
                                      (0, 5), (1, 4), (3, 4), (1, 3), (1, 2), (0, 1)])

    def test_edge_paths(self):
        # Difficult edge cases

        # Edge test1
        edge_test1 = self.build_operands('eb,cb,fb->cef')
        path, path_str = nptest3.einsum_path(*edge_test1, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (0, 2), (0, 1)])

        path, path_str = nptest3.einsum_path(*edge_test1, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (0, 2), (0, 1)])

        # Edge test2
        edge_test2 = self.build_operands('dd,fb,be,cdb->cef')
        path, path_str = nptest3.einsum_path(*edge_test2, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (0, 3), (0, 1), (0, 1)])

        path, path_str = np.einsum_path(*edge_test2, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (0, 3), (0, 1), (0, 1)])

        # Edge test3
        edge_test3 = self.build_operands('bca,cdb,dbf,afc->')
        path, path_str = nptest3.einsum_path(*edge_test3, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])

        path, path_str = nptest3.einsum_path(*edge_test3, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])

        # Edge test4
        edge_test4 = self.build_operands('dcc,fce,ea,dbf->ab')
        path, path_str = nptest3.einsum_path(*edge_test4, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (0, 3), (0, 2), (0, 1)])

        path, path_str = nptest3.einsum_path(*edge_test4, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (1, 2), (0, 2), (0, 1)])

        # Edge test5
        edge_test4 = self.build_operands('a,ac,ab,ad,cd,bd,bc->',
                                         size_dict={"a": 20, "b": 20, "c": 20, "d": 20})
        path, path_str = nptest3.einsum_path(*edge_test4, optimize='greedy')
        self.assert_path_equal(path, ['einsum_path', (0, 1), (0, 1, 2, 3, 4, 5)])

        path, path_str = nptest3.einsum_path(*edge_test4, optimize='optimal')
        self.assert_path_equal(path, ['einsum_path', (0, 1), (0, 1, 2, 3, 4, 5)])

    def test_path_type_input(self):
        # Test explicit path handeling
        path_test = self.build_operands('dcc,fce,ea,dbf->ab')

        path, path_str =  nptest3.einsum_path(*path_test, optimize=False)
        print(path)
        print(path_str)
        self.assert_path_equal(path, ['einsum_path', (0, 1, 2, 3)])

        path, path_str = nptest3.einsum_path(*path_test, optimize=True)
        print(path)
        print(path_str)
        self.assert_path_equal(path, ['einsum_path', (0, 3), (0, 2), (0, 1)])

        exp_path = ['einsum_path', (0, 2), (0, 2), (0, 1)]
        path, path_str = nptest3.einsum_path(*path_test, optimize=exp_path)
        print(path)
        print(path_str)
        self.assert_path_equal(path, exp_path)

        # Double check einsum works on the input path
 


if __name__ == '__main__':
    unittest.main()
