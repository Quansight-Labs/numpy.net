using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NumpyDotNet;
using NumpyLib;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            var matrix = np.arange(0, 100).reshape(new shape(10, 10));

            var viewmatrix = matrix.A(new Slice(2, 7, 1), "2:7:1");         // slice can be either string or Slice() format
            var copymatrix = matrix.A("2:7:1", new Slice(2, 7, 1)).Copy();

            System.Diagnostics.Trace.WriteLine("Full matrix data");
            System.Diagnostics.Trace.WriteLine(matrix);
            System.Diagnostics.Trace.WriteLine("Sub matrix view data");
            System.Diagnostics.Trace.WriteLine(viewmatrix);
            System.Diagnostics.Trace.WriteLine("Copy matrix data");
            System.Diagnostics.Trace.WriteLine(copymatrix);

            // change the data in the view.  Notice the matrix gets changed too, but not the copy.
            viewmatrix[":"] = viewmatrix * 10;                              

            System.Diagnostics.Trace.WriteLine("Full matrix data (with view x10)");
            System.Diagnostics.Trace.WriteLine(matrix);
            System.Diagnostics.Trace.WriteLine("Sub matrix view data (x10)");
            System.Diagnostics.Trace.WriteLine(viewmatrix);
            System.Diagnostics.Trace.WriteLine("copy matrix data (not changed)");
            System.Diagnostics.Trace.WriteLine(copymatrix);

            // change the data in the copy.  Notice the matrix and the view are not impacted.
            copymatrix[":"] = copymatrix * 100;

            System.Diagnostics.Trace.WriteLine("Full matrix data (not changed)");
            System.Diagnostics.Trace.WriteLine(matrix);
            System.Diagnostics.Trace.WriteLine("Sub matrix view data (not changed)");
            System.Diagnostics.Trace.WriteLine(viewmatrix);
            System.Diagnostics.Trace.WriteLine("copy matrix data (x100)");
            System.Diagnostics.Trace.WriteLine(copymatrix);

            Console.ReadLine();

        }
    }
}
