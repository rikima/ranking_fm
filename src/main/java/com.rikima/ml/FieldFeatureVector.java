package com.rikima.ml;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;
import java.util.TreeSet;

public class FieldFeatureVector implements Serializable {
  public static final float DEFAULT_VALUE = 1.0f;

  private int size;

  private int[] fields;
  private int[] features;
  private double[] values;

  public FieldFeatureVector(String hashingFFMFormat) {
    String[] ss = hashingFFMFormat.trim().split(" ");

    TreeSet<FieldFeature> buf = new TreeSet<FieldFeature>();
    for (int i = 0; i < ss.length; ++i) {
      String ffv = ss[i];
      String[] ss2 = ffv.split(":");
      int field = Integer.parseInt(ss2[0]);
      int feature = Integer.parseInt(ss2[1]);
      float value = Float.parseFloat(ss2[2]);
      buf.add(new FieldFeature(field, feature, value));
    }

    this.size = buf.size();
    this.fields = new int[this.size];
    this.features = new int[this.size];
    this.values = new double[this.size];

    int i = 0;
    for (Iterator<FieldFeature> iter = buf.iterator(); iter.hasNext(); ) {
      FieldFeature ff = iter.next();
      this.fields[i] = ff.getFieldId();
      this.features[i] = ff.getFeatureId();
      this.values[i] = ff.getValue();
      i++;
    }
  }

  public FieldFeatureVector(int fieldId, String libsvmFormat) {
    // TODO
    // error handling
    String[] fvs = libsvmFormat.split(Constants.SPACE);
    this.size = fvs.length;

    this.fields = new int[this.size];
    this.features = new int[this.size];
    this.values = new double[this.size];

    for (int i = 0; i < fvs.length; ++i) {
      String fv = fvs[i];
      String[] ss = fv.split(Constants.COLON);
      int feature = Integer.parseInt(ss[0]);
      float value = Float.parseFloat(ss[1]);

      this.fields[i] = fieldId;
      this.features[i] = feature;
      this.values[i] = value;
    }
  }

  public FieldFeatureVector(int fieldId, List<Integer> sortedFeatureIds) {
    // TODO
    // error handling
    this.size = sortedFeatureIds.size();

    this.fields = new int[this.size];
    this.features = new int[this.size];
    this.values = new double[this.size];

    for (int i = 0; i < sortedFeatureIds.size(); ++i) {
      int featureId = sortedFeatureIds.get(i);

      this.fields[i] = fieldId;
      this.features[i] = featureId;
      this.values[i] = DEFAULT_VALUE;
    }
  }


  public FieldFeatureVector(int size) {
    this.size = size;

    this.fields = new int[size];
    this.features = new int[size];
    this.values = new double[size];
  }


  public void normalize() {
    double norm = 0.0;
    for (double v : values) {
      norm += v * v;
    }
    for (int i = 0; i < values.length; ++i) {
      values[i] /= Math.sqrt(norm);
    }
  }

  public int size() {
        return size;
    }

  public String toString() {
    StringBuilder sb = new StringBuilder();
    int prev = -1;
    for (int i = 0; i < this.size(); ++i) {
      assert prev < this.features[i];
      sb.append(String.format(" %d:%d:%f", this.fields[i], this.features[i], this.values[i]));
      prev = this.features[i];
    }
    return sb.toString();
  }

  public FieldFeatureVector setField(int index, int fieldIndex) {
    this.fields[index] = fieldIndex;
    return this;
  }

  public int field(int index) {
        return this.fields[index];
    }

  public FieldFeatureVector setFeature(int index, int featureIndex) {
    this.features[index] = featureIndex;
    return this;
  }

  public int feature(int index) {
        return this.features[index];
    }

  public FieldFeatureVector setValue(int index, double value) {
    this.values[index] = value;
    return this;
  }

  public double value(int index) {
        return this.values[index];
    }

}
