package com.dj.models.mnist;

import org.junit.Ignore;
import org.junit.Test;

public class MnistTrainerTest {

    @Test
    @Ignore("Test will never end since there is no condition to stop training")
    public void testTrainMnist() {
        MnistTrainer.trainMnistNN(true);
    }
}