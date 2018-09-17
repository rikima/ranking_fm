package com.rikima.ml;

import java.io.Serializable;
import java.util.*;

public class FeatureVector implements Serializable, Comparable<FeatureVector> {
  // fields ---------------------------
  private double norm = -1.0;

  private int size;

  private final int[] features;
  private final double[] values;


  // constructors ---------------------
  public FeatureVector(String pureSvmformat) {
    String[] ss = pureSvmformat.trim().split(Constants.SPACE);

    TreeSet<Feature> buf = new TreeSet<Feature>();
    for (int i = 0; i < ss.length; ++i) {
      String ffv = ss[i];
      String[] ss2 = ffv.split(Constants.COLON);
      if (ss2.length == 2) {
        int feature = Integer.parseInt(ss2[0]);
        float value = Float.parseFloat(ss2[1]);
        buf.add(new Feature(feature, value));
      }
    }

    this.size = buf.size();
    this.features = new int[this.size];
    this.values = new double[this.size];

    int i = 0;
    for (Iterator<Feature> iter = buf.iterator(); iter.hasNext(); ) {
      Feature f = iter.next();
      this.features[i] = f.id();
      this.values[i] = f.val();
      i++;
    }
  }

  public FeatureVector(SortedSet<Feature> src) {
    this.size = src.size();
    this.features = new int[src.size()];
    this.values = new double[src.size()];
    int i = 0;
    for (Iterator<Feature> iter = src.iterator(); iter.hasNext(); ) {
      Feature f = iter.next();
      this.features[i] = f.id();
      this.values[i] = f.val();
      i++;
    }
  }

  // methods -------------------------
  public int size() {
      return this.size;
    }

  public Feature getFeature(int index) {
        return new Feature(this.features[index], this.values[index]);
    }

  public int fid(int index) {
        return this.features[index];
    }

  public int findex(int index) {
        return this.features[index] - 1;
    }

  public double value(int index) {
        return this.values[index];
    }

  public double dot(FeatureVector fv) {
    double v = 0;
    int idx1 = 0;
    int idx2 = 0;
    while (idx1 < this.size() && idx2 < fv.size()) {
      int fid1 = this.features[idx1];
      int fid2 = fv.fid(idx2);

      if (fid1 == fid2) {
        v += this.values[idx1] * fv.value(idx2);
        idx1++;
        idx2++;
      } else if (fid1 > fid2) {
        idx2++;
      } else {
        idx1++;
      }
    }
    return v;
  }

  public double getNorm() {
    if (this.norm >= 0) {
      return this.norm;
    }

    double s = 0;
    for (double v : this.values) {
      s += v * v;
    }
    this.norm = s;
    return s;
  }

  public int[] ids() {
        return this.features;
    }

  public double[] values() {
        return this.values;
    }

  public int compareTo(FeatureVector o) {
    int idx1 = 0;
    int idx2 = 0;
    while (idx1 < this.size() && idx2 < o.size()) {
      int fid1 = this.features[idx1];
      int fid2 = o.fid(idx2);

      if (fid1 == fid2) {
        idx1++;
        idx2++;
      } else if (fid1 > fid2) {
        return -1;
      } else {
        return 1;
      }
    }
    return 0;
  }

  public String toString() {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < this.size; ++i) {
      sb.append(String.format("%d:%f ", this.features[i], this.values[i]));
    }
    return sb.toString().trim();
  }
}

