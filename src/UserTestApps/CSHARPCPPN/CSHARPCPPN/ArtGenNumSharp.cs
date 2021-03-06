﻿using System;
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
            Img_Batch = npx.random.standard_normal(new int[] { Batch_Size, width, height, C_Dim });

            Hid_Vec = npx.random.standard_normal(new int[] { Batch_Size, H_Size });

            X_Dat = npx.random.standard_normal(batch_size, width * height, 1);

            Y_Dat = npx.random.standard_normal(batch_size, width * height, 1);

            R_Dat = npx.random.standard_normal(batch_size, width * height, 1);
        }



        public NDArray[] CreateGrid(int width = 32, int height = 32, float scaling = 1.0f)
        {
            Num_Points = width * height;

            var x_range = np.linspace(-1 * scaling, scaling, width);

            var y_range = np.linspace(-1 * scaling, scaling, height);

            var x_mat = np.matmul(np.ones(height, 1), x_range.reshape(1, width));

            var y_mat = np.matmul(y_range.reshape(height, 1), np.ones(1, width));

            var r_mat = np.sqrt((x_mat * x_mat) + (y_mat * y_mat));

            x_mat = npx.tile(x_mat, Batch_Size).reshape(Batch_Size, Num_Points, 1);

            y_mat = npx.tile(y_mat, Batch_Size).reshape(Batch_Size, Num_Points, 1);

            r_mat = npx.tile(r_mat, Batch_Size).reshape(Batch_Size, Num_Points, 1);

            return new NDArray[] { x_mat, y_mat, r_mat };
        }

        public NDArray BuildCPPN(int width, int height, NDArray x_dat, NDArray y_dat, NDArray r_dat, NDArray hid_vec)
        {
            Num_Points = width * height;

            //Scale the hidden vector
            var hid_vec_scaled = np.reshape(hid_vec, new Shape(Batch_Size, 1, H_Size)) * np.ones(new Shape(Num_Points, 1), dtype: np.float32) * Scaling;

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
            var mat = npx.random.standard_normal((int)input.shape[1], out_dim).astype(np.float32);
            var result = np.matmul(input, mat);

            if (with_bias)
            {
                var bias = npx.random.standard_normal(1, out_dim).astype(np.float32);
                result += bias * np.ones(new Shape(input.shape[0], 1), np.float32);
            }

            return result;
        }

        public NDArray TanhSig(int num_layers = 2)
        {
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
                vector = npx.random.uniform(-1, 1, new int[] { Batch_Size, H_Size });
            }
            var data = CreateGrid(width, height, scaling);
            return BuildCPPN(width, height, data[0], data[1], data[2], vector);
        }

        public double[] GenerateArt(int batch_size = 1, int net_size = 16, int h_size = 8, int width = 512, int height = 512, float scaling = 10.0f, bool RGB = true, int? seed = null)
        {
            int? KeyId = null;
            if (seed != null)
            {
                KeyId = seed;
                npx.random.seed(KeyId.GetValueOrDefault());
            }
            else
            {
                KeyId = Math.Abs(DateTime.Now.GetHashCode());
                npx.random.seed(KeyId.GetValueOrDefault());
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

  

    class npx
    {
        public class random
        {
            private static NumpyDotNet.np.random NumpyDotNetRandom { get; set; }

#if true
            public static NDArray standard_normal(params Int32[] newshape)
            {
                var rnddata = NumpyDotNetRandom.standard_normal(new NumpyDotNet.shape(newshape));
                var myarray = np.array(NumpyDotNet.np.AsDoubleArray(rnddata)).reshape(newshape);
                return myarray;
            }

            public static NDArray uniform(int low = 0, int high = 1, params Int32[] newshape)
            {
                var rnddata = NumpyDotNetRandom.uniform(-1, 1, new NumpyDotNet.shape(newshape));
                var myarray = np.array(NumpyDotNet.np.AsFloatArray(rnddata)).reshape(newshape);
                return myarray;
            }

            public static void seed(Int32? seed)
            {
                NumpyDotNetRandom = new NumpyDotNet.np.random();
                NumpyDotNetRandom.seed(seed);
            }
#else
       public static NDArray standard_normal(params Int32[] newshape)
        {
            return np.random.stardard_normal(newshape);
        }

        public static NDArray uniform(int low = 0, int high = 1, params Int32[] newshape)
        {
            return np.random.uniform(low, high, new Shape(newshape));
        }

        public static void seed(Int32? seed)
        {
            if (seed.HasValue)
                np.random.seed(seed.Value);
        }
#endif
        }

        public static NDArray tile(NDArray array, object reps)
        {
            var flattenedArray = array.flatten().ToArray<double>();
            var NDNArray = NumpyDotNet.np.array(flattenedArray);

            var NDBTileArray = NumpyDotNet.np.tile(NDNArray, reps);
            var tiledArray = np.array(NumpyDotNet.np.AsDoubleArray(NDBTileArray));
            return tiledArray;
        }
    }
}