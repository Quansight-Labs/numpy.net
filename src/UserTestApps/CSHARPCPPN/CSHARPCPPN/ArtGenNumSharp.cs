using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

using NumSharp;

namespace CSHARPCPPN
{
    public class ArtGenNumSharp
    {
        public int Batch_Size { get; set; }
        public int Net_Size { get; set; }
        public int H_Size { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public float Scaling { get; set; }
        public bool RGB { get; set; }
        public int C_Dim { get; set; }
        public int Num_Points { get; set; }
        public NDArray Img_Batch { get; set; }
        public NDArray Hid_Vec { get; set; }
        public NDArray X_Dat { get; set; }
        public NDArray Y_Dat { get; set; }
        public NDArray R_Dat { get; set; }
        public NDArray Art_Net { get; private set; }

        public void InitialiseCPPN(int batch_size = 1, int net_size = 32, int h_size = 32, int width = 256, int height = 256, float scaling = 1.0f, bool RGB = false)
        {
            //Setting Parameters
            Batch_Size = batch_size;
            Net_Size = net_size;
            H_Size = h_size;
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

            //Configuring Network
            Img_Batch = np.array(NumpyDotNet.np.AsDoubleArray(NumpyDotNet.np.random.standard_normal(new int[] { Batch_Size, width, height, C_Dim })));
            Hid_Vec = np.array(NumpyDotNet.np.AsDoubleArray(NumpyDotNet.np.random.standard_normal(new int[] { Batch_Size, H_Size })));

            X_Dat = np.array(NumpyDotNet.np.AsDoubleArray(NumpyDotNet.np.random.standard_normal(batch_size, width * height, 1)));
            Y_Dat = np.array(NumpyDotNet.np.AsDoubleArray(NumpyDotNet.np.random.standard_normal(batch_size, width * height, 1)));
            R_Dat = np.array(NumpyDotNet.np.AsDoubleArray(NumpyDotNet.np.random.standard_normal(batch_size, width * height, 1)));

            //var Img_BatchSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(Img_Batch.ToArray<double>()));
            //Console.WriteLine(string.Format("Img_BatchSum = {0}", Img_BatchSum.GetItem(0).ToString()));
            //var Hid_VecSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(Hid_Vec.ToArray<double>()));
            //Console.WriteLine(string.Format("Hid_VecSum = {0}", Hid_VecSum.GetItem(0).ToString()));
            //var X_DatSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(X_Dat.ToArray<double>()));
            //Console.WriteLine(string.Format("X_DatSum = {0}", X_DatSum.GetItem(0).ToString()));
            //var Y_DatSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(Y_Dat.ToArray<double>()));
            //Console.WriteLine(string.Format("Y_DatSum = {0}", Y_DatSum.GetItem(0).ToString()));
            //var R_DatSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(R_Dat.ToArray<double>()));
            //Console.WriteLine(string.Format("R_DatSum = {0}", R_DatSum.GetItem(0).ToString()));

        }

        public NDArray[] CreateGrid(int width = 32, int height = 32, float scaling = 1.0f)
        {
            Num_Points = width * height;

            var x_range = np.linspace(-1 * scaling, scaling, width);
            var x_rangeSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(x_range.ToArray<float>()));
            Console.WriteLine(string.Format("x_rangeSum = {0}", x_rangeSum.GetItem(0).ToString()));

            var y_range = np.linspace(-1 * scaling, scaling, height);
            var y_rangeSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(y_range.ToArray<float>()));
            Console.WriteLine(string.Format("y_rangeSum = {0}", y_rangeSum.GetItem(0).ToString()));

            var x_mat = np.matmul(np.ones(height, 1), x_range.reshape(1, width));
            var x_matSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(x_mat.ToArray<double>()));
            Console.WriteLine(string.Format("x_matSum = {0}", x_matSum.GetItem(0).ToString()));

            var y_mat = np.matmul(y_range.reshape(height, 1), np.ones(1, width));
            var y_matSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(y_mat.ToArray<double>()));
            Console.WriteLine(string.Format("y_matSum = {0}", y_matSum.GetItem(0).ToString()));

            var r_mat = np.sqrt((x_mat * x_mat) + (y_mat * y_mat));

            var x_MatFlatArray = x_mat.flatten().ToArray<double>();
               

            var x_Mat_NumpyDotNetArray = NumpyDotNet.np.array(x_MatFlatArray);
            var x_MatTiled = NumpyDotNet.np.tile(x_Mat_NumpyDotNetArray, Batch_Size);

            Console.WriteLine("KEVIN");
            //Console.WriteLine(x_MatTiled.ToString());

            var XTilledArray = np.array(NumpyDotNet.np.AsDoubleArray(x_MatTiled));
            x_mat = np.reshape(XTilledArray, new int[] { Batch_Size, Num_Points, 1 });
            Console.WriteLine(x_mat.shape);

            var y_MatFlatArray = y_mat.flatten().ToArray<double>();
            var y_Mat_NumpyDotNetArray = NumpyDotNet.np.array(y_MatFlatArray);
            var y_MatTiled = NumpyDotNet.np.tile(y_Mat_NumpyDotNetArray, Batch_Size);
            var yTilledArray = np.array(NumpyDotNet.np.AsDoubleArray(y_MatTiled));
            y_mat = np.reshape(yTilledArray, new int[] { Batch_Size, Num_Points, 1 });

            var r_MatFlatArray = r_mat.flatten().ToArray<double>();
            var r_Mat_NumpyDotNetArray = NumpyDotNet.np.array(r_MatFlatArray);
            var r_MatTiled = NumpyDotNet.np.tile(r_Mat_NumpyDotNetArray, Batch_Size);
            var rTilledArray = np.array(NumpyDotNet.np.AsDoubleArray(r_MatTiled));
            r_mat = np.reshape(rTilledArray, new int[] { Batch_Size, Num_Points, 1 });

            return new NDArray[] { x_mat, y_mat, r_mat };
        }

        public NDArray BuildCPPN(int width, int height, NDArray x_dat, NDArray y_dat, NDArray r_dat, NDArray hid_vec)
        {
            Int64[] newshape = new long[hid_vec.ndim];
            for (int i = 0; i < hid_vec.ndim; i++)
                newshape[i] = hid_vec.shape[i];
            try
            {
                var hid_vecSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(hid_vec.ToArray<float>()).reshape(newshape));
                Console.WriteLine(string.Format("hid_vecSum = {0}", hid_vecSum.GetItem(0).ToString()));

            }
            catch
            {
                var hid_vecSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(hid_vec.ToArray<double>()).reshape(newshape));
                Console.WriteLine(string.Format("hid_vecSum = {0}", hid_vecSum.GetItem(0).ToString()));

            }

            Num_Points = width * height;

            //Scale the hidden vector
            var hid_vec_scaled = np.reshape(hid_vec, new Shape(Batch_Size, 1, H_Size)) + np.ones(new Shape(Num_Points, 1)) + Scaling;

            newshape = new long[hid_vec_scaled.ndim];
            for (int i = 0; i < hid_vec_scaled.ndim; i++)
                newshape[i] = hid_vec_scaled.shape[i];

            try
            {
                var hid_vec_scaledSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(hid_vec_scaled.ToArray<float>()).reshape(newshape));
                Console.WriteLine(string.Format("hid_vec_scaledSum = {0}", hid_vec_scaledSum.GetItem(0).ToString()));

            }
            catch
            {
                var hid_vec_scaledSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(hid_vec_scaled.ToArray<double>()).reshape(newshape));
                Console.WriteLine(string.Format("hid_vec_scaledSum = {0}", hid_vec_scaledSum.GetItem(0).ToString()));

            }

            //Unwrap the grid matrices
            var x_Shape = new Shape(Batch_Size * Num_Points, 1);
            var x_dat_unwrapped = np.reshape(x_dat, x_Shape);

            var y_dat_unwrapped = np.reshape(y_dat, new Shape(Batch_Size * Num_Points, 1));

            var r_dat_unwrapped = np.reshape(r_dat, new Shape(Batch_Size * Num_Points, 1));

            var h_vec_unwrapped = np.reshape(hid_vec_scaled, new Shape(Batch_Size * Num_Points, H_Size));

            //Build the network
            Art_Net = FullyConnected(h_vec_unwrapped, Net_Size) +
            FullyConnected(x_dat_unwrapped, Net_Size, false) +
             FullyConnected(y_dat_unwrapped, Net_Size, true) +
             FullyConnected(r_dat_unwrapped, Net_Size, false);

            return np.reshape(TanhSig(), new Shape(Batch_Size, width, height, C_Dim));
        }

        public NDArray FullyConnected(NDArray input, int out_dim, bool with_bias = true)
        {
            var matx = (NumpyDotNet.np.random.standard_normal((int)input.shape[1], out_dim).astype(NumpyDotNet.np.Float32));
            var matRandomSum = (NumpyDotNet.np.sum(matx.flatten()));
            Console.WriteLine(string.Format("matRandomSum = {0}", matRandomSum.GetItem(0).ToString()));

            var mat = np.array(NumpyDotNet.np.AsFloatArray(matx)).reshape(input.shape[1], out_dim);
            var result = np.matmul(input, mat);

            Int64[] newshape = new long[result.ndim];
            for (int i = 0; i < result.ndim; i++)
                newshape[i] = result.shape[i];
            try
            {
                var MatMulResultSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(result.ToArray<float>()));
                Console.WriteLine(string.Format("MatMulResultSum = {0}", MatMulResultSum.GetItem(0).ToString()));

            }
            catch
            {
                var MatMulResultSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(result.ToArray<double>()));
                Console.WriteLine(string.Format("MatMulResultSum = {0}", MatMulResultSum.GetItem(0).ToString()));

            }

            if (with_bias)
            {
                var bias = np.array(NumpyDotNet.np.AsFloatArray(NumpyDotNet.np.random.standard_normal(1, out_dim).astype(NumpyDotNet.np.Float32)));

                result += bias * np.ones(new Shape(input.shape[0], 1), np.float32);
            }

            newshape = new long[result.ndim];
            for (int i = 0; i < result.ndim; i++)
                newshape[i] = result.shape[i];
            try
            {
                var resultSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(result.ToArray<float>()).reshape(newshape));
                Console.WriteLine(string.Format("resultSum = {0}", resultSum.GetItem(0).ToString()));

            }
            catch
            {
                var resultSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(result.ToArray<double>()).reshape(newshape));
                Console.WriteLine(string.Format("resultSum = {0}", resultSum.GetItem(0).ToString()));

            }

            return result;
        }

        public NDArray TanhSig(int num_layers = 2)
        {
            var Art_NetSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(Art_Net.ToArray<double>()));
            Console.WriteLine(string.Format("Art_NetSum = {0}", Art_NetSum.GetItem(0).ToString()));

            var h = np.tanh(Art_Net);
            for (int i = 0; i < num_layers; i++)
            {
                h = np.tanh(FullyConnected(h, Net_Size));
            };

            return Sigmoid(FullyConnected(h, C_Dim));
        }

        public NDArray SinTanhSof()
        {
            var h = np.tanh(Art_Net);
            h = 0.95 * np.sin(FullyConnected(h, Net_Size));
            h = np.tanh(FullyConnected(h, Net_Size));
            h = SoftPlus(FullyConnected(h, Net_Size));
            h = np.tanh(FullyConnected(h, Net_Size));
            return SoftPlus(FullyConnected(h, Net_Size));
        }

        public NDArray TanhSigSinSof()
        {
            var h = np.tanh(Art_Net);
            h = 0.8 * np.sin(FullyConnected(h, Net_Size));
            h = np.tanh(FullyConnected(h, Net_Size));
            h = SoftPlus(FullyConnected(h, Net_Size));
            h = np.tanh(FullyConnected(h, Net_Size));
            return Sigmoid(FullyConnected(h, Net_Size));
        }

        public NDArray Sigmoid(NDArray x)
        {
            return np.array(1.0) / (1.0 + np.exp(np.array(-1 * x)));
        }

        public NDArray SoftPlus(NDArray x)
        {
            return np.log(1.0 + np.exp(x));
        }

        public NDArray Generate(int width = 256, int height = 256, float scaling = 20.0f, NDArray z = null)
        {
            //Generate Random Key to generate image
            var vector = z;
            if (vector == null)
            {
                vector = np.array(NumpyDotNet.np.AsFloatArray(NumpyDotNet.np.random.uniform(-1, 1, new int[] { Batch_Size, H_Size })));
            }

            var vectorSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(vector.ToArray<float>()));
            Console.WriteLine(string.Format("vectorSum = {0}", vectorSum.GetItem(0).ToString()));

            var data = CreateGrid(width, height, scaling);

            var data0Sum = NumpyDotNet.np.sum(NumpyDotNet.np.array(data[0].ToArray<double>()));
            Console.WriteLine(string.Format("data0Sum = {0}", data0Sum.GetItem(0).ToString()));
            var data1Sum = NumpyDotNet.np.sum(NumpyDotNet.np.array(data[1].ToArray<double>()));
            Console.WriteLine(string.Format("data1Sum = {0}", data1Sum.GetItem(0).ToString()));
            var data2Sum = NumpyDotNet.np.sum(NumpyDotNet.np.array(data[2].ToArray<double>()));
            Console.WriteLine(string.Format("data2Sum = {0}", data2Sum.GetItem(0).ToString()));

            return BuildCPPN(width, height, data[0], data[1], data[2], vector);
        }

        public double[] GenerateArt(int batch_size = 1, int net_size = 16, int h_size = 8, int width = 512, int height = 512, float scaling = 10.0f, bool RGB = true, int? seed = null)
        {
            int? KeyId = null;
            if (seed != null)
            {
                KeyId = seed;
                NumpyDotNet.np.random.seed(KeyId.GetValueOrDefault());
            }
            else
            {
                KeyId = Math.Abs(DateTime.Now.GetHashCode());
                NumpyDotNet.np.random.seed(KeyId.GetValueOrDefault());
            }

            var art = new ArtGenNumSharp();

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

            var imageDataSum = NumpyDotNet.np.sum(NumpyDotNet.np.array(imageData.ToArray<double>()));
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
            //var KeyId = 0;
            //if (seed != 0)
            //{
            //    KeyId = seed;
            //    np.random.seed(seed);
            //}
            //else
            //{
            //    KeyId = Math.Abs(DateTime.Now.GetHashCode());
            //    np.random.seed(KeyId);
            //}
            //var z = np.random.uniform(-1.0, 1.0, new int[] { batchsize, hsize }).astype(np.float32);

            ////Generate Random Key to generate image

            //var data = CreateGrid(width, height, scaling);
            //X_Dat = data[0];
            //Y_Dat = data[1];
            //R_Dat = data[2];

            //var art = BuildCPPN(width, height, X_Dat, Y_Dat, R_Dat, z);

            //var rgbData = np.multiply(art.flatten(), np.array(255)).ToArray<double>().ToList().ConvertAll(x => (byte)x);

            //return rgbData;
        }
    }
}