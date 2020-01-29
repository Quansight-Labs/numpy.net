using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading;
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
using System.Windows.Threading;
using NumpyDotNet;
using NumpyLib;
#if NPY_INTP_64
using npy_intp = System.Int64;
#else
using npy_intp = System.Int32;
#endif


namespace StressTestApp
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private ObservableCollection<string> Logs = new ObservableCollection<string>();

        public MainWindow()
        {
            InitializeComponent();

            LogOutput.ItemsSource = Logs;
        }

        private bool StressTestThreadsRunning = false;

        private void StartTest_Click(object sender, RoutedEventArgs e)
        {
            string sNumThreads = NumThreads.Text;

            int iNumThreads = 1;

            Int32.TryParse(sNumThreads, out iNumThreads);

            StressTestThreadsRunning = true;
            for (int i = 0; i < iNumThreads; i++)
            {
                Task.Run(() => StressTestTask());
            }
        }

        private void StopTest_Click(object sender, RoutedEventArgs e)
        {
            StressTestThreadsRunning = false;
        }

        private void StressTestTask()
        {
            Thread.CurrentThread.Name = "StressTestTaskThread";

            int LoopsRun = 0;
            int LoopsToRun = 20;

            while (StressTestThreadsRunning)
            {

                System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
                sw.Start();

                int AASize = 1600000;
                int AADim1 = 400;

                try
                {
                    var AA = np.arange(AASize, dtype: np.Int32).reshape((AADim1, -1));
                    var BB = np.arange(AASize/AADim1, dtype: np.Int16);


                    var AA1 = AA["1:40:2", "1:-2:3"] as ndarray;

                    for (int i = 0; i < LoopsToRun; i++)
                    {
                        var AA2 = AA / 3;
                        var AA3 = AA2 + i;
                        var AABB = (AA * BB) as ndarray;

                        var indexes = np.where(AABB < 100) as ndarray[];

                        var masked = AABB.ravel()[np.flatnonzero(indexes)] as ndarray;

                    }

                    var output = AA[new Slice(15, 25, 2), new Slice(15, 25, 2)];
                    LoopsRun++;
                }
                catch (Exception ex)
                {
                    string ExMessage = string.Format("stress test threw an exception: {0}", ex.Message);
                    Log(ExMessage);
                }
   

                sw.Stop();

                string LogMessage = string.Format("stress test #{0} took {1} milliseconds\n", LoopsRun, sw.ElapsedMilliseconds);
                Log(LogMessage);
            }

            return;
        }

        private void Log(string LogMessage)
        {
            Application.Current.Dispatcher.BeginInvoke(DispatcherPriority.Background, new Action(() => 
            {
                this.Logs.Add(LogMessage);
            }));
        }

   
    }
}
