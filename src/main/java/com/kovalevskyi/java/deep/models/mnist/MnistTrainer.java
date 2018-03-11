package com.kovalevskyi.java.deep.models.mnist;

import com.google.common.primitives.Ints;
import com.kovalevskyi.java.deep.core.model.activation.Relu;
import com.kovalevskyi.java.deep.core.model.activation.Sigmoid;
import com.kovalevskyi.java.deep.core.model.graph.ConnectedNeuron;
import com.kovalevskyi.java.deep.core.model.graph.InputNeuron;
import com.kovalevskyi.java.deep.core.model.graph.Neuron;
import mnist.MnistReader;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MnistTrainer {

    public static void trainMnistNN(final boolean debug) {
        System.out.println("Downloading MNIst images");
        Downloader.downloadMnist();
        System.out.println("done\n");

        final Random random = new Random();
        final double learningRate = 0.0001;

        System.out.println("Creating network");
        List<Neuron> inputLayer = createLayer(InputNeuron::new, 784);
        List<Neuron> hiddenLayer
                = createLayer(() ->
                    new ConnectedNeuron
                            .Builder()
                            .activationFunction(new Relu())
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
                inputNeuron.connect(hiddenNeuron, random.nextDouble());
            });
        });

        hiddenLayer.stream().forEach(hiddenNeuron -> outputLayer.stream().forEach(outputNeuron -> {
            hiddenNeuron.connect(outputNeuron, random.nextDouble());
        }));
        System.out.println("done\n");

        System.out.println("loading training data in memory");
        final int[] trainLabels = MnistReader.getLabels(Downloader.MNIST_TRAIN_SET_LABELS_FILE.toString());
        final List<int[][]> trainImagesRaw = MnistReader.getImages(Downloader.MNIST_TRAIN_SET_IMAGES_FILE.toString());
        final int[][] trainImages
                = trainImagesRaw.stream().map(Ints::concat).collect(Collectors.toList()).toArray(new int[0][0]);
        System.out.println("done\n");

        System.out.println("loading testing data in memory");
        final int[] testLabels = MnistReader.getLabels(Downloader.MNIST_TEST_SET_LABELS_FILE.toString());
        final List<int[][]> testImagesRaw = MnistReader.getImages(Downloader.MNIST_TEST_SET_IMAGES_FILE.toString());
        final int[][] testImages
                = testImagesRaw.stream().map(Ints::concat).collect(Collectors.toList()).toArray(new int[0][0]);
        System.out.println("done\n");
        
        for (int i =0; i < 1000; i++) {
            System.out.printf("Error on test: %f\n", calculateError(inputLayer, outputLayer, testImages, testLabels));
//            System.out.printf(
//                    "Error on training: %f\n",
//                    calculateError(inputLayer, outputLayer, trainImages, trainLabels));
            System.out.printf("Training epoch #%d\n", i);
            trainIteration(inputLayer, outputLayer, trainImages, trainLabels);
        }
    }
    
    private static void trainIteration(
            final List<Neuron> inputLayer,
            final List<Neuron> outputLayer,
            final int[][] images,
            final int[] labels) {
        for (int imageIndex = 0; imageIndex < images.length; imageIndex++) {
            final int[] image = images[imageIndex];
            IntStream.range(0, image.length).forEach(i ->
                inputLayer.get(i).forwardSignalReceived(null, (double) image[i] / 256.)
            );
            for (int i = 0; i < 10; i++) {
                final double actualValue = ((ConnectedNeuron)outputLayer.get(i)).getForwardResult();
                final double expectedResult = labels[imageIndex] == i ? 1.0 : 0.0;
                outputLayer.get(i).backwardSignalReceived(2. * (expectedResult - actualValue));
            }
        }
        System.out.println();
    }

    private static double calculateError(
            final List<Neuron> inputLayer,
            final List<Neuron> outputLayer,
            final int[][] images,
            final int[] labels) {
        List<Double> errors = new ArrayList<>(images.length);
        for (int imageIndex = 0; imageIndex < images.length; imageIndex++) {
            final int[] image = images[imageIndex];
            for (int i = 0; i < image.length; i++) {
                inputLayer.get(i).forwardSignalReceived(null, (double) image[i] / 256.);
            }
            int answer = 0;
            double probability = -1.;
            int expectedValue = -1;
            for (int i = 0; i < 10; i++) {
                final double actualValue = ((ConnectedNeuron)outputLayer.get(i)).getForwardResult();
                if (actualValue > probability) {
                    probability = actualValue;
                    answer = i;
                }
                if (labels[imageIndex] == i) expectedValue = i;
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
