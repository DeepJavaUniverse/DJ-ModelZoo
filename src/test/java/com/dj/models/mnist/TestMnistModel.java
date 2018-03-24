package com.dj.models.mnist;

import com.dj.core.model.graph.Neuron;
import com.dj.core.serializer.ModelWrapper;
import com.dj.core.serializer.SerializerHelper;
import org.junit.Test;

import java.util.List;
import java.util.stream.IntStream;

import static junit.framework.TestCase.assertTrue;

public class TestMnistModel {

    @Test
    public void testInference() {
        // Downloading MNIst images
        MnistDownloader.downloadMnist();

        // loading testing data in memory
        final int[] testLabelsRaw = MnistReader.getLabels(MnistDownloader.MNIST_TEST_SET_LABELS_FILE.toString());
        final double[][] testLabels = new double[testLabelsRaw.length][];
        IntStream.range(0, testLabelsRaw.length).forEach(i -> testLabels[i] = MnistTrainer.convertLabel(testLabelsRaw[i]));
        final List<int[][]> testImagesRaw
                = MnistReader.getImages(MnistDownloader.MNIST_TEST_SET_IMAGES_FILE.toString());
        final double[][] testImages = new double[testImagesRaw.size()][];
        IntStream.range(0, testImagesRaw.size())
                .forEach(i -> testImages[i] = MnistTrainer.convertImageToTheInput(testImagesRaw.get(i)));

        String path = MnistTrainer.class.getClassLoader().getResource("com/dj/models/mnist/mnist.dj").getPath()
                .toString();
        ModelWrapper modelWrapper = SerializerHelper.deserializeFromFile(path);
        List<Neuron> inputLayer = modelWrapper.getInputLayer();
        List<Neuron> outputLayer = modelWrapper.getOutputLayer();
        double error = MnistTrainer.calculateError(inputLayer, outputLayer, testImages, testLabels);
        assertTrue(error < .2);
    }
}
