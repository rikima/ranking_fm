package com.rikima.ml.fm;

import org.junit.Test;
import static junit.framework.TestCase.assertTrue;

import java.io.File;
import java.io.FileReader;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.List;

import static com.rikima.ml.util.MQRankingData.shuffle;
import com.rikima.ml.eval.Evaluation;
import com.rikima.ml.LabeledFeatureVector;

/**
 * Created by a14350 on 2017/01/15.
 */
public class FMTrainerTest {

    @Test
    public void testTrain() throws Exception {
        int K = 50;
        double lambda = 0.0001;
        int numTry = 20;

        FMTrainer trainer = new FMTrainer(lambda);
        File trainFile = new File("./data/sample/a9a.svmformat");

        FileReader fr = new FileReader(trainFile);
        LineNumberReader lnr = new LineNumberReader(fr);

        String line = null;

        List<LabeledFeatureVector> trainData = new ArrayList<LabeledFeatureVector>();
        while ((line = lnr.readLine()) != null) {
            LabeledFeatureVector lfv = new LabeledFeatureVector(line);
            assertTrue(lfv.y() == 1 || lfv.y() == -1);
            trainData.add(lfv);
        }
        shuffle(trainData);

        System.out.println("K: " + K);
        int featureDimension = 125;

        double logloss = 0;
        int c = 0;

        trainer.init(K, featureDimension);
        long t = 0;
        for (int epoch = 0; epoch < numTry; ++epoch) {
            for (int i = 0; i < trainData.size(); ++i) {
                long tt = System.currentTimeMillis();
                LabeledFeatureVector lfv = trainData.get(i);
                trainer.train(lfv);
                tt = System.currentTimeMillis() - tt;
                logloss += trainer.logloss(lfv);
                t += tt;
                c += 1;
            }
            System.out.println("#" + epoch + " logloss:" + logloss/c);
        }


        double mem = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024.0 / 1024.0;

        System.out.println("mem: " + mem + " [MB]");
        System.out.println("#train data: " + c + " process time / example: " + (float) t / c + " [ms]");

        System.out.println("total time: " + t + " [ms]");

        // closed test
        Evaluation eval = new Evaluation();
        for (int i = 0; i < trainData.size(); ++i) {
            long tt = System.currentTimeMillis();
            LabeledFeatureVector lfv = trainData.get(i);

            double score = trainer.score(lfv);
            int ay;
            if (score > 0) {
                ay = 1;
            } else {
                ay = -1;
            }
            eval.setResult(lfv.y(), ay, score);
        }
        eval.printResult("closed test");

        // open test
        eval = new Evaluation();
        File testFile = new File("./data/sample/a9a.t.svmformat");
        fr = new FileReader(testFile);
        lnr = new LineNumberReader(fr);

        line = null;
        List<LabeledFeatureVector> testData = new ArrayList<LabeledFeatureVector>();
        while ((line = lnr.readLine()) != null) {
            LabeledFeatureVector lfv = new LabeledFeatureVector(line);
            assertTrue(lfv.y() == 1 || lfv.y() == -1);
            testData.add(lfv);
        }

        for (LabeledFeatureVector lfv : testData) {
            assertTrue(lfv.y() == 1 || lfv.y() == -1);
            double score = trainer.score(lfv);
            int ay;
            if (score > 0) {
                ay = 1;
            } else {
                ay = -1;
            }
            eval.setResult(lfv.y(), ay, score);
        }
        eval.printResult("open test");
    }
}
