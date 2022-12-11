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
        public static np.random random { get; set; }

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

            Img_Batch = random.standard_normal(new shape( batchSize, width, height, C_Dim ));

            Hid_Vec = random.standard_normal(new shape( batchSize, hSize ));

            X_Dat = random.standard_normal(new shape(batchSize, width * height, 1));
            Y_Dat = random.standard_normal(new shape(batchSize, width * height, 1));
            R_Dat = random.standard_normal(new shape(batchSize, width * height, 1));
        }

        public List<ndarray> CreateGrid(int width = 32, int height = 32, float scaling = 1.0f)
        {
            Num_Points = width * height;

            double ret_step = 0;
            ndarray x_range = np.linspace(-1 * scaling, scaling, ref ret_step, width);

            ndarray y_range = np.linspace(-1 * scaling, scaling, ref ret_step, height);

            ndarray x_mat = np.matmul(np.ones(new shape(height, 1)), x_range.reshape(1, width));

            ndarray y_mat = np.matmul(y_range.reshape(height, 1), np.ones(new shape(1, width)));

            ndarray r_mat = np.sqrt((x_mat * x_mat) + (y_mat * y_mat));

            x_mat = np.tile(x_mat.flatten(), BatchSize).reshape(BatchSize, Num_Points, 1);
            y_mat = np.tile(y_mat.flatten(), BatchSize).reshape(BatchSize, Num_Points, 1);
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
            Num_Points = width * height;
            // Scale the hidden vector
            ndarray hid_vec_scaled = np.reshape(hid_vec, new shape(BatchSize, 1, HSize)) * np.ones((Num_Points, 1), dtype: np.Float32) * Scaling;

            //Unwrap the grid matrices

            ndarray x_dat_unwrapped = np.reshape(x_dat, new shape(BatchSize * Num_Points, 1));

            ndarray y_dat_unwrapped = np.reshape(y_dat, new shape(BatchSize * Num_Points, 1));

            ndarray r_dat_unwrapped = np.reshape(r_dat, new shape(BatchSize * Num_Points, 1));

            ndarray h_vec_unwrapped = np.reshape(hid_vec_scaled, new shape(BatchSize * Num_Points, HSize));

            //Build the network
            Art_Net = FullyConnected(h_vec_unwrapped, NetSize) +
                FullyConnected(x_dat_unwrapped, NetSize, false) +
                FullyConnected(y_dat_unwrapped, NetSize, true) +
                FullyConnected(r_dat_unwrapped, NetSize, false);

            //Set Activation function
            var output = TanhSig();

            var model = np.reshape(output, new shape(BatchSize, width, height, C_Dim));
            return model;
        }

        public ndarray TanhSig(int numOfLayers = 2)
        {
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
            var mat = random.standard_normal(new shape(input.shape.iDims[1], out_dim)).astype(np.Float32);

            var result = np.matmul(input, mat);

            if (with_bias)
            {
                var bias = random.standard_normal(new shape(1, out_dim)).astype(np.Float32);
                result += bias * np.ones(new shape(input.shape.iDims[0], 1), np.Float32);
            }

            return result;
        }

        public ndarray Sigmoid(ndarray x)
        {
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
                vector = random.uniform(-1, 1, new shape( BatchSize, HSize )).astype(np.Float32);
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
                random = new np.random();
                random.seed(KeyId);
            }
            else
            {
                KeyId = Math.Abs(DateTime.Now.GetHashCode());
                random = new np.random();
                random.seed(KeyId);
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

            var imgData = np.array(np.array(1) - imageData);
            if (C_Dim > 1)
            {
                imgData = np.array(imgData.reshape((height, width, C_Dim)) * 255.0);
            }
            else
            {
                imgData = np.array(imgData.reshape((height, width)) * 255.0);
            }

            return (double[])imgData.ToArray();
        }
    }
}