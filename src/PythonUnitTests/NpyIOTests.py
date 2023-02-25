import unittest
import numpy as np
import sys
from nptest import nptest

MAGIC_PREFIX = b'\x93NUMPY'
MAGIC_LEN = len(MAGIC_PREFIX) + 2
ZIP_PREFIX = b'PK\x03\x04'

class NpyIOTests(unittest.TestCase):

    def test_save_and_load(self):

        t1 = np.arange(1,7);
        np.save('c:/temp/t1', t1)
        kev = np.load('c:/temp/t1.npy')

        t2 = t1.reshape(2,3)
        np.save('c:/temp/t2', t2)
        kev = np.load('c:/temp/t2.npy')

        t3 = t2.reshape(3,2)
        np.save('c:/temp/t3', t3)
        kev = np.load('c:/temp/t3.npy')

 

    def test_load_struct_test(self):

        fid = open('c:/temp/t3.npy', "rb")


        N = len(MAGIC_PREFIX)
        magic = fid.read(N)
        # If the file size is less than N, we need to make sure not
        # to seek past the beginning of the file
        fid.seek(-min(N, len(magic)), 1)  # back-up

        version = read_magic(fid)

        import struct

        if version == (1, 0):
            hlength_type = '<H'
        elif version == (2, 0):
            hlength_type = '<I'
        else:
            raise ValueError("Invalid version %r" % version)

        hlength_str = _read_bytes(fid, struct.calcsize(hlength_type), "array header length")

        kevin = struct.unpack(hlength_type, hlength_str)
        header_length = struct.unpack(hlength_type, hlength_str)[0]
        header = _read_bytes(fid, header_length, "array header")

        return

def read_magic(fp):
        """ Read the magic string to get the version of the file format.

        Parameters
        ----------
        fp : filelike object

        Returns
        -------
        major : int
        minor : int
        """
        magic_str = _read_bytes(fp, MAGIC_LEN, "magic string")
        if magic_str[:-2] != MAGIC_PREFIX:
            msg = "the magic string is not correct; expected %r, got %r"
            raise ValueError(msg % (MAGIC_PREFIX, magic_str[:-2]))
        if sys.version_info[0] < 3:
            major, minor = map(ord, magic_str[-2:])
        else:
            major, minor = magic_str[-2:]
        return major, minor


def _read_bytes(fp, size, error_template="ran out of data"):
        """
        Read from file-like object until size bytes are read.
        Raises ValueError if not EOF is encountered before size bytes are read.
        Non-blocking objects only supported if they derive from io objects.

        Required as e.g. ZipExtFile in python 2.6 can return less data than
        requested.
        """
        data = bytes()
        while True:
            # io files (default in python3) return None or raise on
            # would-block, python2 file will truncate, probably nothing can be
            # done about that.  note that regular files can't be non-blocking
            try:
                r = fp.read(size - len(data))
                data += r
                if len(r) == 0 or len(data) == size:
                    break
            except io.BlockingIOError:
                pass
        if len(data) != size:
            msg = "EOF: reading %s, expected %d bytes got %d"
            raise ValueError(msg % (error_template, size, len(data)))
        else:
            return data

if __name__ == '__main__':
    unittest.main()
