package com.kovalevskyi.java.deep.models.mnist;

import org.junit.Test;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertFalse;

public class DownloaderTest {
    @Test
    public void name() {
        Downloader.clearDownloadedFiles();
        assertFalse(Downloader.MNIST_TRAIN_SET_IMAGES_FILE_PATH.exists());
        assertFalse(Downloader.MNIST_TRAIN_SET_LABELS_FILE_PATH.exists());
        assertFalse(Downloader.MNIST_TEST_SET_IMAGES_FILE_PATH.exists());
        assertFalse(Downloader.MNIST_TEST_SET_LABELS_FILE_PATH.exists());
        Downloader.downloadMnist();
        assertTrue(Downloader.MNIST_TRAIN_SET_IMAGES_FILE_PATH.exists());
        assertTrue(Downloader.MNIST_TRAIN_SET_LABELS_FILE_PATH.exists());
        assertTrue(Downloader.MNIST_TEST_SET_IMAGES_FILE_PATH.exists());
        assertTrue(Downloader.MNIST_TEST_SET_LABELS_FILE_PATH.exists());
        Downloader.clearDownloadedFiles();
        assertFalse(Downloader.MNIST_TRAIN_SET_IMAGES_FILE_PATH.exists());
        assertFalse(Downloader.MNIST_TRAIN_SET_LABELS_FILE_PATH.exists());
        assertFalse(Downloader.MNIST_TEST_SET_IMAGES_FILE_PATH.exists());
        assertFalse(Downloader.MNIST_TEST_SET_LABELS_FILE_PATH.exists());
    }
}
