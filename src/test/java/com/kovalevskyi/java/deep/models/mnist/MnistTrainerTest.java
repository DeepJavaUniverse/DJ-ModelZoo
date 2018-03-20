package com.kovalevskyi.java.deep.models.mnist;

import org.junit.Ignore;
import org.junit.Test;

import static org.junit.Assert.*;

public class MnistTrainerTest {

    @Test
    @Ignore("Test will never end since there is no condition to stop training")
    public void testTrainMnist() {
        MnistTrainer.trainMnistNN(true);
    }
}