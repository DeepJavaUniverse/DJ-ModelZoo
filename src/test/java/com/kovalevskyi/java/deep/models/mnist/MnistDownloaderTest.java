package com.kovalevskyi.java.deep.models.mnist;

import org.junit.Test;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertFalse;

public class MnistDownloaderTest {
    
    @Test
    public void name() {
        MnistDownloader.clearDownloadedFiles();
        assertFalse(MnistDownloader.MNIST_TRAIN_SET_IMAGES_FILE.exists());
        assertFalse(MnistDownloader.MNIST_TRAIN_SET_LABELS_FILE.exists());
        assertFalse(MnistDownloader.MNIST_TEST_SET_IMAGES_FILE.exists());
        assertFalse(MnistDownloader.MNIST_TEST_SET_LABELS_FILE.exists());
        MnistDownloader.downloadMnist();
        assertTrue(MnistDownloader.MNIST_TRAIN_SET_IMAGES_FILE.exists());
        assertTrue(MnistDownloader.MNIST_TRAIN_SET_LABELS_FILE.exists());
        assertTrue(MnistDownloader.MNIST_TEST_SET_IMAGES_FILE.exists());
        assertTrue(MnistDownloader.MNIST_TEST_SET_LABELS_FILE.exists());
        MnistDownloader.clearDownloadedFiles();
        assertFalse(MnistDownloader.MNIST_TRAIN_SET_IMAGES_FILE.exists());
        assertFalse(MnistDownloader.MNIST_TRAIN_SET_LABELS_FILE.exists());
        assertFalse(MnistDownloader.MNIST_TEST_SET_IMAGES_FILE.exists());
        assertFalse(MnistDownloader.MNIST_TEST_SET_LABELS_FILE.exists());
    }
}
