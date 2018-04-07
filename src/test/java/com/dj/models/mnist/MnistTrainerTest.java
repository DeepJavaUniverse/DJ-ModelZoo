package com.dj.models.mnist;

import com.dj.core.serializer.ModelWrapper;
import com.dj.core.serializer.SerializerHelper;
import org.junit.Ignore;
import org.junit.Test;

public class MnistTrainerTest {

    @Test
    @Ignore("Test will never end since there is no condition to stop training")
    public void testTrainMnist() {
        MnistTrainer.downloadDataAndtrainMnistNN(true);
    }

    @Test
    @Ignore("Test will never end since there is no condition to stop training")
    public void testTrainKaggleMnist() {
        MnistTrainer.trainMnistNNOnKaggleData(true);
    }

    @Test
    @Ignore("Test will never end since there is no condition to stop training")
    public void testTrainMnistContinue() {
        String path = MnistTrainer.class.getClassLoader().getResource("com/dj/models/mnist/mnist.dj").getPath()
                .toString();
        ModelWrapper modelWrapper = SerializerHelper.deserializeFromFile(path);
        MnistTrainer.downloadDataAndtrainMnistNN(modelWrapper);
    }
}