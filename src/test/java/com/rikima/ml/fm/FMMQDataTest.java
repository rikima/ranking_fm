package com.rikima.ml.fm;

import com.rikima.ml.eval.Evaluation;
import com.rikima.ml.LabeledFeatureVector;
import org.junit.Test;

import java.io.File;
import java.util.*;

import static com.rikima.ml.util.MQRankingData.loadMQData;
import static com.rikima.ml.util.MQRankingData.loadMQRankingData;
import static org.junit.Assert.*;

public class FMMQDataTest {

    @Test
    public void testLoadMQRankingData() throws Exception {
        int featureDimensionMax = 100;
        File file = new File("./data/MQ2007/Fold1/train.txt");

        Map<String, TreeSet<LabeledFeatureVector>> data = loadMQRankingData(file, featureDimensionMax);

        for (Iterator<String> iter = data.keySet().iterator(); iter.hasNext(); ) {
            String query = iter.next();
            System.out.println("---");
            System.out.println("query: " + query);
            TreeSet<LabeledFeatureVector> ranking = data.get(query);

            int i = 0;
            for (LabeledFeatureVector lfv : ranking) {
                i++;
                System.out.println(" #" + i + " " + lfv.toString());
            }
        }
    }


    @Test
    public void testTrainFM4MQ2007() throws Exception {
        int K = 50;
        double lambda = 0.0001;
        int numTry = 20;
        int featureDimensionMax = 100;
        int featureDimension = 200;

        FMTrainer trainer = new FMTrainer(lambda);
        File trainFile = new File("./data/MQ2007/Fold1/train.txt");

        List<LabeledFeatureVector> trainData = loadMQData(trainFile, featureDimensionMax);
        System.out.println("K: " + K);

        double logloss = 0;
        int c = 0;

        trainer.init(K, featureDimension);
        long t = 0;
        for (int epoch = 0; epoch < numTry; ++epoch) {
            for (int i = 0; i < trainData.size(); ++i) {
                long tt = System.currentTimeMillis();
                LabeledFeatureVector lfv = trainData.get(i);
                // TODO
                // ranking learning
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
        //
        {
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
        }

        /**
         * open test
         */
        {
            Evaluation eval = new Evaluation();
            File testFile = new File("./data/MQ2007/Fold1/test.txt");
            List<LabeledFeatureVector> testData = loadMQData(testFile, featureDimensionMax);
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


    @Test
    public void testTrainFM4MQ2008() throws Exception {
        int K = 50;
        double lambda = 0.0001;
        int numTry = 20;
        int featureDimensionMax = 100;
        int featureDimension = 200;

        FMTrainer trainer = new FMTrainer(lambda);
        File trainFile = new File("./data/MQ2008/Fold1/train.txt");
        List<LabeledFeatureVector> trainData = loadMQData(trainFile, featureDimensionMax);
        System.out.println("K: " + K);

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

        /**
         * open test
         */
        File testFile = new File("./data/MQ2008/Fold1/test.txt");
        List<LabeledFeatureVector> testData = loadMQData(testFile, featureDimensionMax);
        eval = new Evaluation();
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
