using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using NumpyDotNet;



namespace WPFApp1
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            var matrix = np.arange(0, 100).reshape(new shape(10, 10));

            var viewmatrix = matrix.A(new Slice(2, 7, 1), "2:7:1");         // slice can be either string or Slice() format
            var copymatrix = matrix.A("2:7:1", new Slice(2, 7, 1)).Copy();

            MatrixDG1.ItemsSource = GetDataTable(matrix).DefaultView;
            ViewDG1.ItemsSource = GetDataTable(viewmatrix).DefaultView;
            CopyDG1.ItemsSource = GetDataTable(copymatrix).DefaultView;


            // change the data in the view.  Notice the matrix gets changed too, but not the copy.
            viewmatrix[":"] = viewmatrix * 10;

            MatrixDG2.ItemsSource = GetDataTable(matrix).DefaultView;
            ViewDG2.ItemsSource = GetDataTable(viewmatrix).DefaultView;
            CopyDG2.ItemsSource = GetDataTable(copymatrix).DefaultView;

            // change the data in the copy.  Notice the matrix and the view are not impacted.
            copymatrix[":"] = copymatrix * 100;

            MatrixDG3.ItemsSource = GetDataTable(matrix).DefaultView;
            ViewDG3.ItemsSource = GetDataTable(viewmatrix).DefaultView;
            CopyDG3.ItemsSource = GetDataTable(copymatrix).DefaultView;


        }

        DataTable GetDataTable(ndarray matrix)
        {
            if (matrix.ndim != 2)
            {
                throw new Exception("We only handle 2D arrays");
            }

            Int32[] RawData = matrix.AsInt32Array();
            int r = 0;

            DataTable dt = new DataTable();

            for (int i = 0; i < matrix.dims[0]; i++)
            {
                dt.Columns.Add();
            }

            for (int i = 0; i < matrix.dims[1]; i++)
            {
                var newRow = dt.NewRow();
                for (int j = 0; j < matrix.dims[0]; j++)
                {
                    newRow[j] = RawData[r];
                    r++;
                }
                dt.Rows.Add(newRow);
            }
            dt.AcceptChanges();
            return dt;


        }

        private void Window_Unloaded(object sender, RoutedEventArgs e)
        {

        }
    }
}
