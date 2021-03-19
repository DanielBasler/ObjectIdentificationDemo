using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ObjectIdentificationDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            var data = mlContext.Data.LoadFromTextFile<ImageData>(@"ObjectData.csv", hasHeader: false);

            var pipeline = mlContext.Transforms
                .LoadImages(
                outputColumnName: "input",
                imageFolder: "objectImages",
                inputColumnName: nameof(ImageData.Path))
                .Append(mlContext.Transforms.ResizeImages(
                outputColumnName: "input",
                imageWidth: 224,
                imageHeight: 224,
                inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(
                outputColumnName: "input",
                interleavePixelColors: true,
                offsetImage: 117))
                .Append(mlContext.Model.LoadTensorFlowModel("tensorflow_inception_graph.pb")
                .ScoreTensorFlowModel(
                outputColumnNames: new[] { "softmax2" },
                inputColumnNames: new[] { "input" },
                addBatchDimensionInput: true));


            Console.WriteLine("Auf die Pipleine wird Fit aufgerufen...");
            var model = pipeline.Fit(data);
            Console.WriteLine("Fit ausgeführt!");

            var engine = mlContext.Model.CreatePredictionEngine<ImageData, ObjectPrediction>(model);
            var labels = File.ReadAllLines(@"c:/temp/ObjectLabel.txt");

            Console.WriteLine("Vorhersage (Prognose der Objekterkennung)...");
            var images = ImageData.ReadDataFromCsv(@"ObjectData.csv");
            foreach (var image in images)
            {
                Console.Write($"  [{image.Path}]: ");
                var prediction = engine.Predict(image).PredictedLabels;
                                
                var i = 0;
                var best = (from p in prediction
                            select new { Index = i++, Prediction = p }).OrderByDescending(p => p.Prediction).First();
                var predictedLabel = labels[best.Index];
                
                Console.WriteLine($"{predictedLabel} {(predictedLabel != image.Label ? "Objekt konnte nicht erkannt werden" : "")}");
            }

        }

        public class ImageData
        {
            [LoadColumn(0)] public string Path;
            [LoadColumn(1)] public string Label;

            public static IEnumerable<ImageData> ReadDataFromCsv(string file)
            {
                return File.ReadAllLines(file)
                    .Select(x => x.Split(';'))
                    .Select(x => new ImageData
                    {
                        Path = x[0],
                        Label = x[1]
                    });
            }
        }

        public class ObjectPrediction
        {
            [ColumnName("softmax2")]
            public float[] PredictedLabels;
        }
    }
}
