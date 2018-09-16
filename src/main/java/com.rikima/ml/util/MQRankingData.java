package com.rikima.ml.util;

import com.rikima.ml.Feature;
import com.rikima.ml.LabeledFeatureVector;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.*;


public class MQRankingData {

  public static void shuffle(List<LabeledFeatureVector> data) {
    int len = data.size();
    Random rand = new Random();
    for (int i = 0; i < (len-1); ++i) {
      int j = rand.nextInt(len - i - 1);

      LabeledFeatureVector f = data.get(i);
      LabeledFeatureVector f2 = data.get(j + i);

      data.set(j + i, f);
      data.set(i, f2);
    }
  }

  public static List<LabeledFeatureVector> loadMQData(File file, int featureDimensionMax) throws IOException {
    LineNumberReader lnr = new LineNumberReader(new FileReader(file));
    List<LabeledFeatureVector> data = new ArrayList<LabeledFeatureVector>();

    SortedSet<Feature> buf = new TreeSet<Feature>();
    String line = null;
    while ((line = lnr.readLine()) != null) {
      int p = line.indexOf("#");
      if (p > 0) {
        line = line.substring(0, p);
      }
      buf.clear();

      String[] ss = line.split(" ");
      // y
      int y = Integer.parseInt(ss[0]);

      // query
      int qid = HashingUtil.hash(ss[1].trim(), featureDimensionMax, 1) + featureDimensionMax;
      Feature f = new Feature(qid, 1.0f);
      buf.add(f);
      for (int i = 2; i < ss.length; ++i) {
        String s = ss[i];
        String[] ss2 = s.split(":");
        int fid = Integer.parseInt(ss2[0]);
        float v = Float.parseFloat(ss2[1]);
        if (v > 0) {
          f = new Feature(fid, v);
          buf.add(f);
        }
      }

      if (y > 0) {
        y = 1;
      } else {
        y = -1;
      }

      LabeledFeatureVector lfv = new LabeledFeatureVector(y, buf);
      data.add(lfv);
      //System.out.println(lfv.toString());
    }
    shuffle(data);

    return data;
  }


  public static Map<String, TreeSet<LabeledFeatureVector>> loadMQRankingData(File file, int featureDimensionMax) throws IOException {
    LineNumberReader lnr = new LineNumberReader(new FileReader(file));

    Map<String, TreeSet<LabeledFeatureVector>> data = new HashMap<String, TreeSet<LabeledFeatureVector>>();

    SortedSet<Feature> buf = new TreeSet<Feature>();
    String line = null;
    while ((line = lnr.readLine()) != null) {
      int p = line.indexOf("#");
      if (p > 0) {
        line = line.substring(0, p);
      }
      buf.clear();

      String[] ss = line.split(" ");
      // y
      int y = Integer.parseInt(ss[0]);

      // query
      String query = ss[1].trim();

      if (!data.containsKey(query)) {
        data.put(query, new TreeSet<LabeledFeatureVector>());
      }
      TreeSet<LabeledFeatureVector> ranking = data.get(query);

      int qid = HashingUtil.hash(ss[1].trim(), featureDimensionMax, 1) + featureDimensionMax;
      Feature f = new Feature(qid, 1.0f);
      buf.add(f);
      for (int i = 2; i < ss.length; ++i) {
        String s = ss[i];
        String[] ss2 = s.split(":");
        int fid = Integer.parseInt(ss2[0]);
        float v = Float.parseFloat(ss2[1]);
        if (v > 0) {
          f = new Feature(fid, v);
          buf.add(f);
        }
      }

      LabeledFeatureVector lfv = new LabeledFeatureVector(y, buf);
      ranking.add(lfv);
    }

    return data;
  }

}
