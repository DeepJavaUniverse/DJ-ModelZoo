package com.dj.models.mnist;


import com.dj.core.helpers.NormalizationHelper;
import com.dj.core.model.activation.LeakyRelu;
import com.dj.core.model.activation.Sigmoid;
import com.dj.core.model.graph.ConnectedNeuron;
import com.dj.core.model.graph.Context;
import com.dj.core.model.graph.InputNeuron;
import com.dj.core.model.graph.Neuron;
import com.dj.core.model.loss.Loss;
import com.dj.core.model.loss.QuadraticLoss;
import com.dj.core.optimizer.Optimizer;
import com.dj.core.optimizer.OptimizerProgressListener;
import com.dj.core.optimizer.SGDOptimizer;
import com.dj.core.serializer.ModelWrapper;
import com.dj.core.serializer.SerializerHelper;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MnistTrainer {

    private static final int INDEX_OF_KAGGLE_IMAGES = 0;

    private static final int INDEX_OF_KAGGLE_LABELS = 1;

    public static void downloadDataAndTrainMnistNN(final boolean debug) {
        final ModelWrapper modelWrapper = createTheModel(debug);
        downloadDataAndTrainMnistNN(modelWrapper);
    }

    public static void downloadDataAndTrainMnistNN(final ModelWrapper modelWrapper) {
        System.out.println("Downloading MNIst images");
        MnistDownloader.downloadMnist();
        System.out.println("done\n");

        System.out.println("loading training data in memory");
        final double[][] trainLabels = loadLabels(MnistDownloader.MNIST_TRAIN_SET_LABELS_FILE.toString());

        final double[][] trainImages = loadImages(MnistDownloader.MNIST_TRAIN_SET_IMAGES_FILE.toString());
        System.out.println("done\n");

        System.out.println("loading testing data in memory");
        final double[][] testLabels = loadLabels(MnistDownloader.MNIST_TEST_SET_LABELS_FILE.toString());

        final double[][] testImages = loadImages(MnistDownloader.MNIST_TEST_SET_IMAGES_FILE.toString());
        System.out.println("done\n");
        trainMnistNN(modelWrapper, trainLabels, trainImages, testLabels, testImages);
    }

    public static void trainMnistNNOnKaggleData(final boolean debug) {
        final ModelWrapper modelWrapper = createTheModel(debug);

        System.out.println("Downloading MNIst images");
        MnistDownloader.downloadMnist();
        System.out.println("done\n");

        System.out.println("Preparing training data");
        final String path = MnistTrainer.class.getClassLoader()
                .getResource("com/dj/models/mnist/train.csv")
                .getPath();
        final double[][][] kaggleData = readKaggleDataTraining(path);
        final double[][] trainImages = kaggleData[INDEX_OF_KAGGLE_IMAGES];
        final double[][] trainLabels = kaggleData[INDEX_OF_KAGGLE_LABELS];
        System.out.println("done");

        System.out.println("Loading testing data");
        final double[][] testLabels = loadLabels(MnistDownloader.MNIST_TEST_SET_LABELS_FILE.toString());

        final double[][] testImages = loadImages(MnistDownloader.MNIST_TEST_SET_IMAGES_FILE.toString());
        System.out.println("done\n");
        trainMnistNN(modelWrapper, trainLabels, trainImages, testLabels, testImages);
        SerializerHelper.serializeToFile(modelWrapper, "/tmp/mnist_kaggle.dj");
    }

    private static double[][][] readKaggleDataTraining(final String path) {
        final File csvData = new File(path);
        final CSVParser parser;
        final List<CSVRecord> records;
        try {
            parser = CSVParser.parse(csvData, Charset.defaultCharset(), CSVFormat.EXCEL.withHeader());
            records= parser.getRecords();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("CSV parsing failed", e);
        }

        final double[][] trainImages = new double[records.size()][784];
        final double[][] trainLabels = new double[records.size()][10];
        IntStream.range(0, records.size()).forEach(imageIndex ->
                IntStream.range(0, 784).forEach(pixelIndex -> {
                    trainImages[imageIndex][pixelIndex] = Integer.parseInt(records.get(imageIndex).get(pixelIndex + 1));
                    final int label = Integer.parseInt(records.get(imageIndex).get(0));
                    trainLabels[imageIndex][label] = 1.;
                })
        );
        final double[][][] result = new double[2][][];
        result[INDEX_OF_KAGGLE_IMAGES] = NormalizationHelper.normalize(trainImages);
        result[INDEX_OF_KAGGLE_LABELS] = trainLabels;
        return result;
    }

    private static void prepareSubmissionData(final ModelWrapper modelWrapper, final String outpuPath) {
        final String path = MnistTrainer.class.getClassLoader()
                .getResource("com/dj/models/mnist/test.csv")
                .getPath();
        final double[][] testImages = readKaggleTestData(path);

        final StringBuilder outputResult = new StringBuilder();
        outputResult.append("ImageId,Label\n");
        IntStream.range(0, testImages.length).forEach(imageIndex -> {
            final double[] image = testImages[imageIndex];
            IntStream.range(0, image.length).forEach(pixelIndex -> {
                modelWrapper.getInputLayer().get(pixelIndex).forwardSignalReceived(null, image[pixelIndex]);
            });
            int answer = 0;
            double probability = -1.;
            for (int i = 0; i < 10; i++) {
                final double actualValue = (modelWrapper.getOutputLayer().get(i)).getForwardResult();
                if (actualValue > probability) {
                    probability = actualValue;
                    answer = i;
                }
            }
            outputResult.append(String.format("%d,%d\n", imageIndex + 1, answer));
        });
        try(BufferedWriter kaggleResultWriter = new BufferedWriter(new FileWriter(new File(outpuPath)))) {
            kaggleResultWriter.write(outputResult.toString());
            kaggleResultWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("writing Kaggle result failed", e);
        }
    }

    private static double[][] readKaggleTestData(final String path) {
        final File csvData = new File(path);
        final CSVParser parser;
        final List<CSVRecord> records;
        try {
            parser = CSVParser.parse(csvData, Charset.defaultCharset(), CSVFormat.EXCEL.withHeader());
            records= parser.getRecords();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("CSV parsing failed", e);
        }

        final double[][] testImages = new double[records.size()][784];
        IntStream.range(0, records.size()).forEach(imageIndex ->
                IntStream.range(0, 784).forEach(pixelIndex -> {
                    testImages[imageIndex][pixelIndex] = Integer.parseInt(records.get(imageIndex).get(pixelIndex));
                })
        );
        return NormalizationHelper.normalize(testImages);
    }

    private static void trainMnistNN(final ModelWrapper modelWrapper,
                                    final double[][] trainLabels,
                                    final double[][] trainImages,
                                    final double[][] testLabels,
                                    final double[][] testImages) {
        List<Neuron> inputLayer = modelWrapper.getInputLayer();
        List<Neuron> outputLayer
                = modelWrapper.getOutputLayer();
        Context context = modelWrapper.getContext();
        final Loss loss = new QuadraticLoss();
        final Optimizer optimizer
                = new SGDOptimizer(loss, 500, new OptimizerProgressListener() {
            @Override
            public void onProgress(final double v, final int i, final int i1) {
                final double updatedLoss = calculateError(inputLayer, outputLayer, testImages, testLabels);
                System.out.printf(
                        "LOSS: %5f, CorrectLoss: %10f,Epoch: %d of %d\n",
                        v,
                        updatedLoss,
                        i,
                        i1);
                SerializerHelper.serializeToFile(modelWrapper, String.format("/tmp/mnist_model_checkpoint_%d.dj", i));
                prepareSubmissionData(modelWrapper, String.format("/tmp/submission_checkpoint_%d.csv", i));
            }
        }, 2.);
        optimizer.train(context, inputLayer, outputLayer, trainImages, trainLabels, testImages, testLabels);

        final ModelWrapper model = new ModelWrapper.Builder().inputLayer(inputLayer).outputLayer(outputLayer).build();
        SerializerHelper.serializeToFile(model, "/tmp/mnist.dj");
    }

    public static double[] convertLabel(final int label) {
        final double[] labels = new double[10];
        labels[label] = 1.;
        return labels;
    }

    public static double[] convertImageToTheInput(final int[][] image) {
        return Arrays.stream(image)
                .flatMapToInt(row -> Arrays.stream(row))
                .mapToDouble(pixel -> pixel)
                .toArray();
    }

    public static double calculateError(
            final List<Neuron> inputLayer,
            final List<Neuron> outputLayer,
            final double[][] images,
            final double[][] labels) {
        List<Double> errors = new ArrayList<>(images.length);
        for (int imageIndex = 0; imageIndex < images.length; imageIndex++) {
            final double[] image = images[imageIndex];
            for (int i = 0; i < image.length; i++) {
                inputLayer.get(i).forwardSignalReceived(null, image[i]);
            }
            int answer = 0;
            double probability = -1.;
            int expectedValue = -1;
            for (int i = 0; i < 10; i++) {
                final double actualValue = (outputLayer.get(i)).getForwardResult();
                if (actualValue > probability) {
                    probability = actualValue;
                    answer = i;
                }
                if (labels[imageIndex][i] > 0) expectedValue = i;
            }
            if (answer == expectedValue) {
                errors.add(0.);
            } else {
                errors.add(1.);
            }
        }
        return errors.stream().mapToDouble(i -> i).average().getAsDouble();
    }

    private static double[][] loadLabels(final String path) {
        final int[] trainLabelsRaw = MnistReader.getLabels(path);
        final double[][] trainLabels = new double[trainLabelsRaw.length][];
        IntStream.range(0, trainLabels.length).forEach(i -> trainLabels[i] = convertLabel(trainLabelsRaw[i]));
        return trainLabels;
    }

    private static double[][] loadImages(final String path) {
        final List<int[][]> trainImagesRaw = MnistReader.getImages(path);
        final double[][] trainImages = new double[trainImagesRaw.size()][];
        IntStream.range(0, trainImagesRaw.size())
                .forEach(i -> trainImages[i] = convertImageToTheInput(trainImagesRaw.get(i)));
        return NormalizationHelper.normalize(trainImages);
    }

    private static ModelWrapper createTheModel(final boolean debug) {
        final Random random = new Random();
        final double learningRate = 0.0005;
        final Context context = new Context(learningRate, debug);

        System.out.println("Creating network");
        List<Neuron> inputLayer = createLayer(InputNeuron::new, 784);
        List<Neuron> hiddenLayer
                = createLayer(() ->
                        new ConnectedNeuron
                                .Builder()
                                .activationFunction(new LeakyRelu())
                                .context(context)
                                .build(),
                10);
        List<Neuron> outputLayer
                = createLayer(() ->
                        new ConnectedNeuron
                                .Builder()
                                .activationFunction(new Sigmoid(true))
                                .context(context)
                                .build(),
                10);

        inputLayer.stream().forEach(inputNeuron -> {
            hiddenLayer.stream().forEach(hiddenNeuron -> {
                inputNeuron.connect(hiddenNeuron, (random.nextDouble() * 2. - 1.) * Math.sqrt(2. / 784.));
            });
        });

        hiddenLayer.stream().forEach(hiddenNeuron -> outputLayer.stream().forEach(outputNeuron -> {
            hiddenNeuron.connect(outputNeuron, (random.nextDouble() * 2. - 1.) / 10.);
        }));
        System.out.println("done\n");
        return new ModelWrapper.Builder().inputLayer(inputLayer).outputLayer(outputLayer).context(context).build();
    }

    private static List<Neuron> createLayer(final Supplier<Neuron> neuronSupplier, final int layerSize) {
        return IntStream.range(0, layerSize).mapToObj(i -> neuronSupplier.get()).collect(Collectors.toList());
    }

}
