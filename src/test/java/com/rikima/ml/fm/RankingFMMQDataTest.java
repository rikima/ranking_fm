package com.rikima.ml.fm;

import java.io.File;
import java.util.*;

import org.junit.Test;
import static org.junit.Assert.assertEquals;

import com.rikima.ml.eval.RankingEvaluation;
import com.rikima.ml.LabeledFeatureVector;
import static com.rikima.ml.util.MQRankingData.loadMQRankingData;

public class RankingFMMQDataTest {
  int K = 50;
  double lambda = 0.0001;
  int numEpochs = 10;
  int featureDimensionMax = 100;
  int featureDimension = 200;

  int position = 10;

  @Test
  public void testTrainRankingFMByMQ() throws Exception {
    String[] dataSets = {"MQ2007", "MQ2008"};
    for (String dataSet : dataSets) {
      double avgNdcg = 0.0;
      int size = 0;
      for (int i = 1; i <= 5; i++) {
        RankingEvaluation eval = evalRankingFMBByMQ(dataSet, i);
        assertEquals(1, 1);
        avgNdcg += eval.getTotalNDCG(position);
        size += eval.size();
      }
      avgNdcg /= size;
      System.out.printf("# avg. NGCG@%d for %s: %f", position, dataSet, avgNdcg);
    }
  }

  private RankingEvaluation evalRankingFMBByMQ(String dataSet, int foldId) throws Exception {
    RankingFMTrainer trainer = new RankingFMTrainer(lambda);
    File trainFile = new File(String.format("./data/%s/Fold%d/train.txt", dataSet, foldId));

    Map<String, TreeSet<LabeledFeatureVector>> trainData = loadMQRankingData(trainFile, featureDimensionMax);
    System.out.println("K: " + K);

    int n = 5;
    long seed = 1;
    Random rand = new Random(seed);


    trainer.init(K, featureDimension);
    long t = 0;
    int c = 0;
    List<LabeledFeatureVector> rankingList = new ArrayList<LabeledFeatureVector>();
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
      double logloss = 0;

      Set<String> queries = trainData.keySet();

      for (Iterator<String> iter = queries.iterator(); iter.hasNext();) {

        String query = iter.next();

        TreeSet<LabeledFeatureVector> ranking = trainData.get(query);
        rankingList.clear();
        rankingList.addAll(ranking);

        for (int j = 0;j < n; ++j) {
          for (int i = 0; i < rankingList.size(); ++i) {
            LabeledFeatureVector ufv = rankingList.get(i);
            if (ufv.y() < 1) {
              break;
            }
            int bound = rankingList.size() - i - 1;
            if (bound < 1) {
              break;
            }
            int ri = rand.nextInt(bound) + i + 1;
            LabeledFeatureVector lfv = rankingList.get(ri);

            long tt = System.currentTimeMillis();
            trainer.train(ufv, lfv);
            tt = System.currentTimeMillis() - tt;
            t += tt;
            logloss += trainer.logloss(ufv, lfv);
            c++;
          }
        }
      }
      System.out.println("#" + epoch + " logloss:" + logloss/c);
    }

    double mem = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024.0 / 1024.0;

    System.out.println("mem: " + mem + " [MB]");
    System.out.println("#train data: " + c + " process time / example: " + (float) t / c + " [ms]");
    System.out.println("total time: " + t + " [ms]");

    /**
     * open test
     */
    RankingEvaluation eval = new RankingEvaluation();
    {
      System.out.println(String.format("# open test for %s Fold%d", dataSet, foldId ));
      File testFile = new File(String.format("./data/%s/Fold%d/test.txt", dataSet, foldId));
      Map<String, TreeSet<LabeledFeatureVector>> testData = loadMQRankingData(testFile, featureDimensionMax);
      Set<String> queries = testData.keySet();


      for (Iterator<String> iter = queries.iterator(); iter.hasNext(); ) {
        String query = iter.next();

        TreeSet<LabeledFeatureVector> ranking = testData.get(query);
        rankingList.clear();
        rankingList.addAll(ranking);

        eval.prepareResult(query);
        for (int i = 0; i < rankingList.size(); ++i) {

          LabeledFeatureVector lfv = rankingList.get(i);
          long tt = System.currentTimeMillis();
          double score = trainer.score(lfv);
          tt = System.currentTimeMillis() - tt;
          eval.addResult(query, score, lfv);
          t += tt;
        }
      }
      eval.printResult(String.format("# open test for %s Fold%d", dataSet, foldId ), position);
    }
    return eval;
  }


  //@Test
  public void testTrainRankingFM4MQ2007() throws Exception {

    RankingFMTrainer trainer = new RankingFMTrainer(lambda);
    File trainFile = new File("./data/MQ2007/Fold1/train.txt");

    Map<String, TreeSet<LabeledFeatureVector>> trainData = loadMQRankingData(trainFile, featureDimensionMax);
    System.out.println("K: " + K);

    int n = 5;

    long seed = 1;
    Random rand = new Random(seed);
    int tc = 0;

    trainer.init(K, featureDimension);
    long t = 0;
    List<LabeledFeatureVector> rankingList = new ArrayList<LabeledFeatureVector>();
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
      Set<String> queries = trainData.keySet();

      double logloss = 0;
      int c = 0;

      for (Iterator<String> iter = queries.iterator(); iter.hasNext();) {

        String query = iter.next();

        TreeSet<LabeledFeatureVector> ranking = trainData.get(query);
        rankingList.clear();
        rankingList.addAll(ranking);

        for (int j = 0; j < n;++j) {
          for (int i = 0; i < rankingList.size(); ++i) {
            LabeledFeatureVector ufv = rankingList.get(i);
            if (ufv.y() < 1 || rankingList.size() <= 1) {
              break;
            }
            int bound = rankingList.size() - i - 1;
            if (bound < 1) {
              break;
            }
            int ri = rand.nextInt(bound) + i + 1;
            LabeledFeatureVector lfv = rankingList.get(ri);

            long tt = System.currentTimeMillis();
            trainer.train(ufv, lfv);
            tt = System.currentTimeMillis() - tt;
            t += tt;
            logloss += trainer.logloss(ufv, lfv);
            c++;
            tc++;
          }
        }
      }
      System.out.println("#" + epoch + " logloss:" + logloss/c);
    }

    double mem = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024.0 / 1024.0;

    System.out.println("mem: " + mem + " [MB]");
    System.out.println("#train data: " + tc + " process time / example: " + (float) t / tc + " [ms]");
    System.out.println("total time: " + t + " [ms]");

    /**
     * closed test
     */
    {
      System.out.println("# closed test");
      Set<String> queries = trainData.keySet();
      RankingEvaluation eval = new RankingEvaluation();

      for (Iterator<String> iter = queries.iterator(); iter.hasNext(); ) {
        String query = iter.next();

        TreeSet<LabeledFeatureVector> ranking = trainData.get(query);
        rankingList.clear();
        rankingList.addAll(ranking);

        eval.prepareResult(query);
        for (int i = 0; i < rankingList.size(); ++i) {

          LabeledFeatureVector lfv = rankingList.get(i);
          long tt = System.currentTimeMillis();
          double score = trainer.score(lfv);
          tt = System.currentTimeMillis() - tt;
          eval.addResult(query, score, lfv);
          t += tt;
        }
      }
      eval.printResult("closed test", position);
    }

    /**
     * open test
     */
    {
      System.out.println("# open test");
      File testFile = new File("./data/MQ2007/Fold1/test.txt");
      Map<String, TreeSet<LabeledFeatureVector>> testData = loadMQRankingData(testFile, featureDimensionMax);
      Set<String> queries = testData.keySet();
      RankingEvaluation eval = new RankingEvaluation();

      for (Iterator<String> iter = queries.iterator(); iter.hasNext(); ) {
        String query = iter.next();

        TreeSet<LabeledFeatureVector> ranking = testData.get(query);
        rankingList.clear();
        rankingList.addAll(ranking);

        eval.prepareResult(query);
        for (int i = 0; i < rankingList.size(); ++i) {

          LabeledFeatureVector lfv = rankingList.get(i);
          long tt = System.currentTimeMillis();
          double score = trainer.score(lfv);
          tt = System.currentTimeMillis() - tt;
          eval.addResult(query, score, lfv);
          t += tt;
        }
      }
      eval.printResult("open test", position);
    }
  }

}
