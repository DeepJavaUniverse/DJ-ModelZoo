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
import java.text.SimpleDateFormat;
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

    private static final SimpleDateFormat optimiserTimeFormat = new SimpleDateFormat("dd-MM-yyyy HH:mm:ss");

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

        final var batchSize = modelWrapper.getContext().getBatchSize();
        IntStream.range(0, testImages.length / batchSize).forEach(batchIndex -> {
            final var from = batchIndex * batchSize;
            final var imagesBatch = fetchBatch(testImages, from, from + batchSize);

            fillInputLayer(modelWrapper.getInputLayer(), imagesBatch);

            final var answer = newArray(batchSize, 0);
            final var probability = newArray(batchSize, -1.);


            final var outputLayer = modelWrapper.getOutputLayer();
            IntStream.range(0, outputLayer.size()).forEach(i -> {
                final var actualValues = (outputLayer.get(i)).getForwardResult();
                IntStream.range(0, actualValues.length).forEach(batchItemIndex -> {
                    if (actualValues[batchItemIndex] > probability[batchItemIndex]) {
                        probability[batchItemIndex] = actualValues[batchItemIndex];
                        answer[batchItemIndex] = i;
                    }
                });

                IntStream.of(answer).forEach(j -> {
                    outputResult.append(String.format("%d,%d\n", i, answer[j]));
                });
            });
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
            public void onProgress(final double loss, final int epoch, final int epochMax) {
                logEpochStatistics(context, inputLayer, outputLayer, testImages, testLabels, epoch, epochMax, loss);
                SerializerHelper.serializeToFile(modelWrapper, String.format("/tmp/mnist_model_checkpoint_%d.dj", epoch));
                prepareSubmissionData(modelWrapper, String.format("/tmp/submission_checkpoint_%d.csv", epoch));
            }
        }, 2.);
        optimizer.train(context, inputLayer, outputLayer, trainImages, trainLabels, testImages, testLabels);

        final ModelWrapper model = new ModelWrapper.Builder().inputLayer(inputLayer).outputLayer(outputLayer).build();
        SerializerHelper.serializeToFile(model, "/tmp/mnist.dj");
    }

    private static void logEpochStatistics(final Context context,
                                           final List<Neuron> inputLayer, final List<Neuron> outputLayer,
                                           final double[][] testImages, final double[][] testLabels,
                                           final int epoch, final int epochMax, final double loss) {
        final double updatedLoss = calculateError(context, inputLayer, outputLayer, testImages, testLabels);
        System.out.printf(
                "%s | Epoch: %d of %d | LOSS: %5f | CorrectLoss: %10f | \n",
                optimiserTimeFormat.format(System.currentTimeMillis()),
                epoch,
                epochMax,
                loss,
                updatedLoss);
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
            final Context context,
            final List<Neuron> inputLayer,
            final List<Neuron> outputLayer,
            final double[][] images,
            final double[][] labels) {
        List<Double> errors = new ArrayList<>(images.length);
        final var batchSize = context.getBatchSize();
        IntStream.range(0, images.length / batchSize).forEach(batchIndex -> {
            final var from = batchIndex * batchSize;
            final var imagesBatch = fetchBatch(images, from, from + batchSize);
            final var labelsBatch = fetchBatch(labels, from, from + batchSize);

            fillInputLayer(inputLayer, imagesBatch);

            final var answer = newArray(batchSize, 0);
            final var probability = newArray(batchSize, -1.);
            final var expectedValue = newArray(batchSize, -1);

            IntStream.range(0, outputLayer.size()).forEach(i -> {
                final var actualValues = (outputLayer.get(i)).getForwardResult();

                IntStream.range(0, actualValues.length).forEach(batchItemIndex -> {
                    if (actualValues[batchItemIndex] > probability[batchItemIndex]) {
                        probability[batchItemIndex] = actualValues[batchItemIndex];
                        answer[batchItemIndex] = i;
                    }
                    if (labelsBatch[batchItemIndex][i] > 0) expectedValue[batchItemIndex] = i;
                });

            });

            IntStream.range(0, answer.length).forEach(batchItemIndex -> {
                errors.add(answer[batchItemIndex] == expectedValue[batchItemIndex] ? 0. : 1.);
            });
        });
        return errors.stream().mapToDouble(i -> i).average().getAsDouble();
    }

    private static void fillInputLayer(final List<Neuron> inputLayer, final double[][] src) {
        IntStream.range(0, inputLayer.size()).forEach(i -> {
            final var inputNeuronIndex = i;
            double[] inputBatch = IntStream
                    .range(0, src.length)
                    .mapToDouble(imageIndex -> src[imageIndex][inputNeuronIndex])
                    .toArray();
            inputLayer.get(i).forwardSignalReceived(null, inputBatch);
        });
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
        final double learningRate = 0.005;
        final int batchSize = 100;
        final Context context = new Context(learningRate, debug, batchSize);

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

    private static double[][] fetchBatch(final double[][] src, final int from, final int to) {
        return Arrays.copyOfRange(src, from, to);
    }

    private static int[] newArray(final int length, final int initValue) {
        final var array = new int[length];
        Arrays.fill(array, initValue);
        return array;
    }

    private static double[] newArray(final int length, final double initValue) {
        final var array = new double[length];
        Arrays.fill(array, initValue);
        return array;
    }
}
