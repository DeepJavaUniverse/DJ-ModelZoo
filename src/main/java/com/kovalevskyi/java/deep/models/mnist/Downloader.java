package com.kovalevskyi.java.deep.models.mnist;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

public abstract class Downloader {

    private static final String HTTP_PROTOCOL = "http";
    private static final String YANN_LECUN_HOST_NAME = "yann.lecun.com";
    private static final String MNIST_TRAIN_SET_IMAGES = "/exdb/mnist/train-images-idx3-ubyte.gz";
    private static final String MNIST_TRAIN_SET_LABELS = "/exdb/mnist/train-labels-idx1-ubyte.gz";
    private static final String MNIST_TEST_SET_IMAGES = "/exdb/mnist/t10k-images-idx3-ubyte.gz";
    private static final String MNIST_TEST_SET_LABELS = "/exdb/mnist/t10k-labels-idx1-ubyte.gz";
    private static final String TMP_DIR_PATH = System.getProperty("java.io.tmpdir");
    
    private static final File MNIST_TRAIN_SET_IMAGES_ZIP_FILE
            = Paths.get(TMP_DIR_PATH, "train-images.gz").toFile();
    private static final File MNIST_TRAIN_SET_LABELS_ZIP_FILE
            = Paths.get(TMP_DIR_PATH, "train-labels.gz").toFile();
    private static final File MNIST_TEST_SET_IMAGES_ZIP_FILE
            = Paths.get(TMP_DIR_PATH, "test-images.gz").toFile();
    private static final File MNIST_TEST_SET_LABELS_ZIP_FILE
            = Paths.get(TMP_DIR_PATH, "test-labels.gz").toFile();
    
    public static final File MNIST_TRAIN_SET_IMAGES_FILE
            = Paths.get(TMP_DIR_PATH, "train-images").toFile();
    public static final File MNIST_TRAIN_SET_LABELS_FILE
            = Paths.get(TMP_DIR_PATH, "train-labels").toFile();
    public static final File MNIST_TEST_SET_IMAGES_FILE
            = Paths.get(TMP_DIR_PATH, "test-images").toFile();
    public static final File MNIST_TEST_SET_LABELS_FILE
            = Paths.get(TMP_DIR_PATH, "test-labels").toFile();

    private Downloader() { }

    public static void downloadMnist() {
        if (MNIST_TRAIN_SET_IMAGES_FILE.exists() &&
                MNIST_TRAIN_SET_LABELS_FILE.exists() &&
                MNIST_TEST_SET_IMAGES_FILE.exists() &&
                MNIST_TEST_SET_LABELS_FILE.exists()) {
            return;
        }
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
            FileUtils.copyURLToFile(trainSetImages, MNIST_TRAIN_SET_IMAGES_ZIP_FILE);
            FileUtils.copyURLToFile(trainSetLabels, MNIST_TRAIN_SET_LABELS_ZIP_FILE);
            FileUtils.copyURLToFile(testSetImages, MNIST_TEST_SET_IMAGES_ZIP_FILE);
            FileUtils.copyURLToFile(testSetLabels, MNIST_TEST_SET_LABELS_ZIP_FILE);
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Failure to download MNIst DataSet", e);
        }

        unZipFile(MNIST_TRAIN_SET_IMAGES_ZIP_FILE, MNIST_TRAIN_SET_IMAGES_FILE);
        unZipFile(MNIST_TRAIN_SET_LABELS_ZIP_FILE, MNIST_TRAIN_SET_LABELS_FILE);
        unZipFile(MNIST_TEST_SET_IMAGES_ZIP_FILE, MNIST_TEST_SET_IMAGES_FILE);
        unZipFile(MNIST_TEST_SET_LABELS_ZIP_FILE, MNIST_TEST_SET_LABELS_FILE);
    }

    public static void clearDownloadedFiles() {
        removeFilesIfExist(
                MNIST_TRAIN_SET_IMAGES_ZIP_FILE,
                MNIST_TRAIN_SET_LABELS_ZIP_FILE,
                MNIST_TEST_SET_IMAGES_ZIP_FILE,
                MNIST_TEST_SET_LABELS_ZIP_FILE,
                MNIST_TRAIN_SET_IMAGES_FILE,
                MNIST_TRAIN_SET_LABELS_FILE,
                MNIST_TEST_SET_IMAGES_FILE,
                MNIST_TEST_SET_LABELS_FILE);
    }

    private static void removeFilesIfExist(final File... files) {
       Stream.of(files).filter(File::exists).forEach(File::delete);
    }

    private static void unZipFile(final File fileToUnzip, final File dest) {
        final byte[] buffer = new byte[1024];

        try(GZIPInputStream src = new GZIPInputStream(new FileInputStream(fileToUnzip));
            FileOutputStream dst = new FileOutputStream(dest)) {

            int len;
            while ((len = src.read(buffer)) > 0) {
                dst.write(buffer, 0, len);
            }
        } catch(IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Unzip process have failed", e);
        }
    }
}
