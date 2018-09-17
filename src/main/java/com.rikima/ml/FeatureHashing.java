package com.rikima.ml;

import com.rikima.ml.util.HashingUtil;

import java.io.*;
import java.util.TreeSet;

public class FeatureHashing implements Serializable {
  public static final int NUM_FEATURES = 1 << 24;
  public static final int SEED = 1;

  private TreeSet<FieldFeature> buf = new TreeSet<FieldFeature>();
  private StringBuilder sb = new StringBuilder();

  public String getHashingSvmformat(String libsvmFormat, boolean isOnehotVector) throws RuntimeException {
    sb.delete(0, sb.length());
    buf.clear();

    String[] ss = libsvmFormat.trim().split(Constants.SPACE);
    for (int i = 0; i < ss.length;++i) {
      String[] ss2 = ss[i].split(Constants.COLON);
      try {
        int fid = HashingUtil.hash(ss2[0], NUM_FEATURES, SEED);
        float value = Float.parseFloat(ss2[1]);

        buf.add(new FieldFeature(0, fid, value));
      } catch (Throwable e) {
        throw new RuntimeException(String.format("parse error: %s", libsvmFormat));
      }
    }

    for (FieldFeature ff : buf) {
      if (isOnehotVector) {
        sb.append(String.format("%d:1 ", ff.getFeatureId()));
      } else {
        sb.append(String.format("%d:%f ", ff.getFeatureId(), ff.getValue()));
      }
    }
    return sb.toString().trim();
  }
}
