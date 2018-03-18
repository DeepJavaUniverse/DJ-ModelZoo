package com.kovalevskyi.java.deep.models.mnist;

import com.google.common.primitives.Ints;
import com.kovalevskyi.java.deep.core.model.activation.LeakyRelu;
import com.kovalevskyi.java.deep.core.model.activation.Relu;
import com.kovalevskyi.java.deep.core.model.activation.Sigmoid;
import com.kovalevskyi.java.deep.core.model.graph.ConnectedNeuron;
import com.kovalevskyi.java.deep.core.model.graph.InputNeuron;
import com.kovalevskyi.java.deep.core.model.graph.Neuron;
import com.kovalevskyi.java.deep.core.model.loss.Loss;
import com.kovalevskyi.java.deep.core.model.loss.QuadraticLoss;
import com.kovalevskyi.java.deep.core.optimizer.Optimizer;
import com.kovalevskyi.java.deep.core.optimizer.OptimizerProgressListener;
import com.kovalevskyi.java.deep.core.optimizer.SGDOptimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class MnistTrainer {

    public static void trainMnistNN(final boolean debug) {
        System.out.println("Downloading MNIst images");
        MnistDownloader.downloadMnist();
        System.out.println("done\n");

        final Random random = new Random();
        final double learningRate = 0.0005;

        System.out.println("Creating network");
        List<Neuron> inputLayer = createLayer(InputNeuron::new, 784);
        List<Neuron> hiddenLayer
                = createLayer(() ->
                    new ConnectedNeuron
                            .Builder()
                            .activationFunction(new LeakyRelu())
                            .debug(debug)
                            .learningRate(learningRate)
                            .build(),
                    10);
        List<Neuron> outputLayer
                = createLayer(() ->
                    new ConnectedNeuron
                            .Builder()
                            .activationFunction(new Sigmoid(true))
                            .debug(debug)
                            .learningRate(learningRate)
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

        System.out.println("loading training data in memory");
        final int[] trainLabelsRaw = MnistReader.getLabels(MnistDownloader.MNIST_TRAIN_SET_LABELS_FILE.toString());
        final double[][] trainLabels = new double[trainLabelsRaw.length][];
        IntStream.range(0, trainLabels.length).forEach(i -> trainLabels[i] = convertLabel(trainLabelsRaw[i]));

        final List<int[][]> trainImagesRaw = MnistReader.getImages(MnistDownloader.MNIST_TRAIN_SET_IMAGES_FILE.toString());
        final double[][] trainImages = new double[trainImagesRaw.size()][];
        IntStream.range(0, trainImagesRaw.size())
                .forEach(i -> trainImages[i] = convertImageToTheInput(trainImagesRaw.get(i)));
        System.out.println("done\n");

        System.out.println("loading testing data in memory");
        final int[] testLabelsRaw = MnistReader.getLabels(MnistDownloader.MNIST_TEST_SET_LABELS_FILE.toString());
        final double[][] testLabels = new double[testLabelsRaw.length][];
        IntStream.range(0, testLabelsRaw.length).forEach(i -> testLabels[i] = convertLabel(testLabelsRaw[i]));
        final List<int[][]> testImagesRaw
                = MnistReader.getImages(MnistDownloader.MNIST_TEST_SET_IMAGES_FILE.toString());
        final double[][] testImages = new double[testImagesRaw.size()][];
        IntStream.range(0, testImagesRaw.size())
                .forEach(i -> testImages[i] = convertImageToTheInput(testImagesRaw.get(i)));
        System.out.println("done\n");

        final Loss loss = new QuadraticLoss();
        final Optimizer optimizer
                = new SGDOptimizer(loss, 100, new OptimizerProgressListener() {
            @Override
            public void onProgress(final double v, final int i, final int i1) {
                final double updatedLoss = calculateError(inputLayer, outputLayer, testImages, testLabels);
                System.out.printf(
                        "LOSS: %5f, CorrectLoss: %10f,Epoch: %d of %d\n",
                        v,
                        updatedLoss,
                        i,
                        i1);
            }
        });
        optimizer.train(inputLayer, outputLayer, trainImages, trainLabels, testImages, testLabels);
    }

    private static double[] convertLabel(final int label) {
        final double[] labels = new double[10];
        labels[label] = 1.;
        return labels;
    }

    private static double[] convertImageToTheInput(final int[][] image) {
        return Arrays.stream(image)
                .flatMapToInt(row -> Arrays.stream(row))
                .mapToDouble(pixel -> ((double)pixel - 128.) / 128.)
                .toArray();
    }

    private static double calculateError(
            final List<Neuron> inputLayer,
            final List<Neuron> outputLayer,
            final double[][] images,
            final double[][] labels) {
        List<Double> errors = new ArrayList<>(images.length);
        for (int imageIndex = 0; imageIndex < images.length; imageIndex++) {
            final double[] image = images[imageIndex];
            for (int i = 0; i < image.length; i++) {
                inputLayer.get(i).forwardSignalReceived(null, (image[i] - 128.) / 128.);
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
                if (labels[imageIndex][i] == 1.0) expectedValue = i;
            }
            if (answer == expectedValue) {
                errors.add(0.);
            } else {
                errors.add(1.);
            }
        }
        return errors.stream().mapToDouble(i -> i).average().getAsDouble();
    }

    private static List<Neuron> createLayer(final Supplier<Neuron> neuronSupplier, final int layerSize) {
        return IntStream.range(0, layerSize).mapToObj(i -> neuronSupplier.get()).collect(Collectors.toList());
    }

}
