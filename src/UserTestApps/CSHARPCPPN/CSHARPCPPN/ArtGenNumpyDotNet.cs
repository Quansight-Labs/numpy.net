using System;
using System.Collections.Generic;
using System.Text;
using NumpyDotNet;

namespace CSHARPCPPN
{
    public class ArtGenNumpyDotNet
    {
        public int BatchSize { get; set; }
        public int NetSize { get; set; }
        public int HSize { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public float Scaling { get; set; }
        public bool RGB { get; set; }
        public int C_Dim { get; set; }
        public int Num_Points { get; set; }
        public ndarray Img_Batch { get; set; }
        public ndarray Hid_Vec { get; set; }
        public ndarray X_Dat { get; set; }
        public ndarray Y_Dat { get; set; }
        public ndarray R_Dat { get; set; }
        public ndarray Art_Net { get; private set; }

        public void InitialiseCPPN(int batchSize = 1, int netSize = 32, int hSize = 32, int width = 256, int height = 256, float scaling = 1.0f, bool RGB = false)
        {
            //Setting Parameters

            BatchSize = batchSize;
            NetSize = netSize;
            HSize = hSize;
            Width = width;
            Height = height;
            Scaling = scaling;
            this.RGB = RGB;

            if (RGB)
            {
                C_Dim = 3;
            }
            else
            {
                C_Dim = 1;
            }

            Num_Points = width * height;

            // Configuring Network

            Img_Batch = np.random.standard_normal(new int[] { batchSize, width, height, C_Dim });

            Hid_Vec = np.random.standard_normal(new int[] { batchSize, hSize });

            X_Dat = np.random.standard_normal(batchSize, width * height, 1);
            Y_Dat = np.random.standard_normal(batchSize, width * height, 1);
            R_Dat = np.random.standard_normal(batchSize, width * height, 1);

            //var Img_BatchSum = np.sum(Img_Batch);
            //Console.WriteLine(string.Format("Img_BatchSum = {0}", Img_BatchSum.GetItem(0).ToString()));
            //var Hid_VecSum = np.sum(Hid_Vec);
            //Console.WriteLine(string.Format("Hid_VecSum = {0}", Hid_VecSum.GetItem(0).ToString()));
            //var X_DatSum = np.sum(X_Dat);
            //Console.WriteLine(string.Format("X_DatSum = {0}", X_DatSum.GetItem(0).ToString()));
            //var Y_DatSum = np.sum(Y_Dat);
            //Console.WriteLine(string.Format("Y_DatSum = {0}", Y_DatSum.GetItem(0).ToString()));
            //var R_DatSum = np.sum(R_Dat);
            //Console.WriteLine(string.Format("R_DatSum = {0}", R_DatSum.GetItem(0).ToString()));
        }

        public List<ndarray> CreateGrid(int width = 32, int height = 32, float scaling = 1.0f)
        {
            Num_Points = width * height;

            double ret_step = 0;
            ndarray x_range = np.linspace(-1 * scaling, scaling, ref ret_step, width, dtype: np.Float32);
            var x_rangeSum = np.sum(x_range);
            Console.WriteLine(string.Format("x_rangeSum = {0}", x_rangeSum.GetItem(0).ToString()));

            ndarray y_range = np.linspace(-1 * scaling, scaling, ref ret_step, height, dtype: np.Float32);
            var y_rangeSum = np.sum(y_range);
            Console.WriteLine(string.Format("y_rangeSum = {0}", y_rangeSum.GetItem(0).ToString()));

            ndarray x_mat = np.matmul(np.ones(new shape(height, 1)), x_range.reshape(1, width));
            var x_matSum = np.sum(x_mat);
            Console.WriteLine(string.Format("x_matSum = {0}", x_matSum.GetItem(0).ToString()));

            ndarray y_mat = np.matmul(y_range.reshape(height, 1), np.ones(new shape(1, width)));
            var y_matSum = np.sum(y_mat);
            Console.WriteLine(string.Format("y_matSum = {0}", y_matSum.GetItem(0).ToString()));

            ndarray r_mat = np.sqrt((x_mat * x_mat) + (y_mat * y_mat));

            var x_matFlatten = x_mat.flatten();
            var x_matFlattenSum = np.sum(x_matFlatten);
            Console.WriteLine(string.Format("x_matFlattenSum = {0}", x_matFlattenSum.GetItem(0).ToString()));

            x_mat = np.tile(x_mat.flatten(), BatchSize);
            Console.WriteLine("KEVIN");
            //Console.WriteLine(x_mat.ToString());

            x_mat = np.reshape(x_mat, new shape ( BatchSize, Num_Points, 1 ));
            Console.WriteLine(x_mat.shape);

            //var x_matSum2 = np.sum(x_mat);
            //Console.WriteLine(string.Format("x_matSum2 = {0}", x_matSum2.GetItem(0).ToString()));
            y_mat = np.tile(y_mat.flatten(), BatchSize).reshape(BatchSize, Num_Points, 1);
            //var y_matSum2 = np.sum(y_mat);
            //Console.WriteLine(string.Format("y_matSum2 = {0}", y_matSum2.GetItem(0).ToString()));

            r_mat = np.tile(r_mat.flatten(), BatchSize).reshape(BatchSize, Num_Points, 1);

            return new List<ndarray>
            {
                x_mat,
                y_mat,
                r_mat
            };
        }

        public ndarray BuildCPPN(int width, int height, ndarray x_dat, ndarray y_dat, ndarray r_dat, ndarray hid_vec)
        {
            var hid_vecSum = np.sum(hid_vec);
            Console.WriteLine(string.Format("hid_vecSum = {0}", hid_vecSum.GetItem(0).ToString()));

            Num_Points = width * height;
            // Scale the hidden vector
            ndarray hid_vec_scaled = np.reshape(hid_vec, new shape(BatchSize, 1, HSize)) + np.ones(new shape(Num_Points, 1)) + Scaling;
            var hid_vec_scaledSUM = np.sum(hid_vec_scaled);
            Console.WriteLine(string.Format("hid_vec_scaledSUM = {0}", hid_vec_scaledSUM.GetItem(0).ToString()));


            //Unwrap the grid matrices

            ndarray x_dat_unwrapped = np.reshape(x_dat, new shape(BatchSize * Num_Points, 1));

            ndarray y_dat_unwrapped = np.reshape(y_dat, new shape(BatchSize * Num_Points, 1));

            ndarray r_dat_unwrapped = np.reshape(r_dat, new shape(BatchSize * Num_Points, 1));

            ndarray h_vec_unwrapped = np.reshape(hid_vec_scaled, new shape(BatchSize * Num_Points, HSize));

            //Build the network
            var k1 = FullyConnected(h_vec_unwrapped, NetSize);
            var k1Sum = np.sum(k1.flatten());
            Console.WriteLine(string.Format("k1Sum = {0}", k1Sum.GetItem(0).ToString()));

            var k2 = FullyConnected(x_dat_unwrapped, NetSize, false);
            var k2Sum = np.sum(k2.flatten());
            Console.WriteLine(string.Format("k2Sum = {0}", k2Sum.GetItem(0).ToString()));

            var k3 = FullyConnected(y_dat_unwrapped, NetSize, true);
            var k3Sum = np.sum(k3.flatten());
            Console.WriteLine(string.Format("k3Sum = {0}", k3Sum.GetItem(0).ToString()));

            var k4 = FullyConnected(r_dat_unwrapped, NetSize, false);
            var k4Sum = np.sum(k4.flatten());
            Console.WriteLine(string.Format("k4Sum = {0}", k4Sum.GetItem(0).ToString()));

            Art_Net = k1 + k2 + k3 + k4;

            //Set Activation function
            var output = TanhSig();

            var model = np.reshape(output, new shape(BatchSize, width, height, C_Dim));
            return model;
        }

        public ndarray TanhSig(int numOfLayers = 2)
        {
            var Art_NetSum = np.sum(Art_Net.flatten());
            Console.WriteLine(string.Format("Art_NetSum = {0}", Art_NetSum.GetItem(0).ToString()));

            var h = np.tanh(Art_Net);
            for (int i = 0; i < numOfLayers; i++)
            {
                h = np.tanh(FullyConnected(h, NetSize));
            };

            return Sigmoid(FullyConnected(h, C_Dim));
        }

        public ndarray SinTanhSof()
        {
            var h = np.tanh(Art_Net);
            h = 0.95 * np.sin(FullyConnected(h, NetSize));
            h = np.tanh(FullyConnected(h, NetSize));
            h = SoftPlus(FullyConnected(h, NetSize));
            h = np.tanh(FullyConnected(h, NetSize));
            return SoftPlus(FullyConnected(h, NetSize));
        }

        public ndarray TanhSigSinSof()
        {
            var h = np.tanh(Art_Net);
            h = 0.8 * np.sin(FullyConnected(h, NetSize));
            h = np.tanh(FullyConnected(h, NetSize));
            h = SoftPlus(FullyConnected(h, NetSize));
            h = np.tanh(FullyConnected(h, NetSize));
            return Sigmoid(FullyConnected(h, NetSize));
        }

        public ndarray FullyConnected(ndarray input, int out_dim, bool with_bias = true)
        {
            var mat = np.random.standard_normal((int)input.shape.iDims[1], out_dim).astype(np.Float32);
            //var matRandomSum = np.sum(mat.flatten());
            //Console.WriteLine(string.Format("matRandomSum = {0}", matRandomSum.GetItem(0).ToString()));


            var result = np.matmul(input, mat);

            //var MatMulResultSum = np.sum(result.flatten());
            //Console.WriteLine(string.Format("MatMulResultSum = {0}", MatMulResultSum.GetItem(0).ToString()));


            if (with_bias)
            {
                var bias = np.random.standard_normal(1, out_dim).astype(np.Float32);
                result += bias * np.ones(new shape(input.shape.iDims[0], 1), np.Int32);
            }
            var resultSum = np.sum(result);
            Console.WriteLine(string.Format("resultSum = {0}", resultSum.GetItem(0).ToString()));
            return result;
        }

        public ndarray Sigmoid(ndarray x)
        {
            var SigmoidSum = np.sum(x);
            Console.WriteLine(string.Format("SigmoidSum = {0}", SigmoidSum.GetItem(0).ToString()));
            return (np.array(1.0) / (1.0 + np.exp(-1 * x))) as ndarray;
        }

        public ndarray SoftPlus(ndarray x)
        {
            return np.log(1.0 + np.exp(x));
        }

        public ndarray Generate(int width = 256, int height = 256, float scaling = 20.0f, ndarray z = null)
        {
            //Generate Random Key to generate image
            var vector = z;
            if (vector == null)
            {
                vector = np.random.uniform(-1, 1, new int[] { BatchSize, HSize }).astype(np.Float32);
            }

            //var vectorSum = np.sum(vector);
            //Console.WriteLine(string.Format("vectorSum = {0}", vectorSum.GetItem(0).ToString()));

            var data = CreateGrid(width, height, scaling);

            //var data0Sum = np.sum(data[0]);
            //Console.WriteLine(string.Format("data0Sum = {0}", data0Sum.GetItem(0).ToString()));
            //var data1Sum = np.sum(data[1]);
            //Console.WriteLine(string.Format("data1Sum = {0}", data1Sum.GetItem(0).ToString()));
            //var data2Sum = np.sum(data[2]);
            //Console.WriteLine(string.Format("data2Sum = {0}", data2Sum.GetItem(0).ToString()));

            return BuildCPPN(width, height, data[0], data[1], data[2], vector);
        }

        public double[] GenerateArt(int batch_size = 1, int net_size = 16, int h_size = 8, int width = 512, int height = 512, float scaling = 10.0f, bool RGB = true, int? seed = null)
        {
            int? KeyId = null;
            if (seed != null)
            {
                KeyId = seed;
                np.random.seed(KeyId);
            }
            else
            {
                KeyId = Math.Abs(DateTime.Now.GetHashCode());
                np.random.seed(KeyId);
            }

            var art = new ArtGenNumpyDotNet();

            art.InitialiseCPPN(batch_size, net_size, RGB: RGB);

            if (RGB)
            {
                C_Dim = 3;
            }
            else
            {
                C_Dim = 1;
            }

            var imageData = art.Generate(width, height, scaling);
            var imageDataSum = np.sum(imageData);
            Console.WriteLine(string.Format("imageDataSum = {0}", imageDataSum.GetItem(0).ToString()));

            var imgData = np.array(1 - imageData);
            if (C_Dim > 1)
            {
                imgData = np.array(imgData.reshape((height, width, C_Dim)) * 255.0);
            }
            else
            {
                imgData = np.array(imgData.reshape((height, width)) * 255.0);
            }

            return imgData.ToArray<double>();
        }
    }
}