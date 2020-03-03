using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace CSHARPCPPN

{
    internal class Program
    {
        private static void Main(string[] args)
        {
            //GenerateImageNumpyDotNet();
            GenerateImageNumSharp();
        }

        public static void GenerateImageNumpyDotNet(int xwidth = 256, int yheight = 256)
        {
            var art = new ArtGenNumpyDotNet();
            var barray = art.GenerateArt(batch_size: 1, net_size: 8, h_size: 16, width: xwidth, height: yheight, scaling: 5.0f, RGB: false, seed: 040692);
            Image<Rgba32> image = new Image<Rgba32>(xwidth, yheight);

            var Count = 0;

            for (int y = 0; y < xwidth; y++)
            {
                for (int x = 0; x < yheight; x++)
                {
                    Rgba32 pixel = image[x, y];
                    pixel.R = (byte)barray[Count];
                    pixel.G = (byte)barray[Count];
                    pixel.B = (byte)barray[Count];
                    pixel.A = 255;
                    image[x, y] = pixel;
                    Count++;
                }
            }

            image.Save(@"C:\temp\NumpyTestImg.jpg");
        }

        public static void GenerateImageNumSharp(int xwidth = 256, int yheight = 256)
        {
            var art = new ArtGenNumSharp();
            var barray = art.GenerateArt(batch_size: 1, net_size: 8, h_size: 16, width: xwidth, height: yheight, scaling: 5.0f, RGB: false, seed: 040692);
            Image<Rgba32> image = new Image<Rgba32>(xwidth, yheight);

            var Count = 0;

            for (int y = 0; y < xwidth; y++)
            {
                for (int x = 0; x < yheight; x++)
                {
                    Rgba32 pixel = image[x, y];
                    pixel.R = (byte)barray[Count];
                    pixel.G = (byte)barray[Count];
                    pixel.B = (byte)barray[Count];
                    pixel.A = 255;
                    image[x, y] = pixel;
                    Count++;
                }
            }

            image.Save(@"C:\temp\NumSharpTestImg.jpg");
        }
    }
}