package com.kovalevskyi.java.deep.models.mnist;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Paths;

public abstract class Downloader {

    private static final String HTTP_PROTOCOL = "http";
    private static final String YANN_LECUN_HOST_NAME = "yann.lecun.com";
    private static final String MNIST_TRAIN_SET_IMAGES = "/exdb/mnist/train-images-idx3-ubyte.gz";
    private static final String MNIST_TRAIN_SET_LABELS = "/exdb/mnist/train-labels-idx1-ubyte.gz";
    private static final String MNIST_TEST_SET_IMAGES = "/exdb/mnist/t10k-images-idx3-ubyte.gz";
    private static final String MNIST_TEST_SET_LABELS = "/exdb/mnist/t10k-labels-idx1-ubyte.gz";
    private static final String TMP_DIR_PATH = System.getProperty("java.io.tmpdir");
    
    public static final File MNIST_TRAIN_SET_IMAGES_FILE_PATH
            = Paths.get(TMP_DIR_PATH, "train-images.gz").toFile();
    public static final File MNIST_TRAIN_SET_LABELS_FILE_PATH
            = Paths.get(TMP_DIR_PATH, "train-labels.gz").toFile();
    public static final File MNIST_TEST_SET_IMAGES_FILE_PATH
            = Paths.get(TMP_DIR_PATH, "test-images.gz").toFile();
    public static final File MNIST_TEST_SET_LABELS_FILE_PATH
            = Paths.get(TMP_DIR_PATH, "test-images.gz").toFile();

    private Downloader() { }

    public static void downloadMnist() {
        final URL trainSetImages;
        final URL trainSetLabels;
        final URL testSetImages;
        final URL testSetLabels;
        try {
            trainSetImages = new URL(HTTP_PROTOCOL, YANN_LECUN_HOST_NAME, MNIST_TRAIN_SET_IMAGES);
            trainSetLabels= new URL(HTTP_PROTOCOL, YANN_LECUN_HOST_NAME, MNIST_TRAIN_SET_LABELS);
            testSetImages = new URL(HTTP_PROTOCOL, YANN_LECUN_HOST_NAME, MNIST_TEST_SET_IMAGES);
            testSetLabels= new URL(HTTP_PROTOCOL, YANN_LECUN_HOST_NAME, MNIST_TEST_SET_LABELS);
        } catch (MalformedURLException e) {
            e.printStackTrace();
            throw new RuntimeException("Failure to create URLs that are required to download MNIst DataSet", e);
        }
        try {
            FileUtils.copyURLToFile(trainSetImages, MNIST_TRAIN_SET_IMAGES_FILE_PATH);
            FileUtils.copyURLToFile(trainSetLabels, MNIST_TRAIN_SET_LABELS_FILE_PATH);
            FileUtils.copyURLToFile(testSetImages, MNIST_TEST_SET_IMAGES_FILE_PATH);
            FileUtils.copyURLToFile(testSetLabels, MNIST_TEST_SET_LABELS_FILE_PATH);
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Failure to download MNIst DataSet", e);
        }
    }

    public static void clearDownloadedFiles() {
        if (MNIST_TRAIN_SET_IMAGES_FILE_PATH.exists()) {
            MNIST_TRAIN_SET_IMAGES_FILE_PATH.delete();
        }
        if (MNIST_TRAIN_SET_LABELS_FILE_PATH.exists()) {
            MNIST_TRAIN_SET_LABELS_FILE_PATH.delete();
        }
        if (MNIST_TEST_SET_IMAGES_FILE_PATH.exists()) {
            MNIST_TEST_SET_IMAGES_FILE_PATH.delete();
        }
        if (MNIST_TEST_SET_LABELS_FILE_PATH.exists()) {
            MNIST_TEST_SET_LABELS_FILE_PATH.delete();
        }
    }
}
