package com.rikima.ml;

import java.io.Serializable;
import java.util.SortedSet;

/**
 * Created by a14350 on 2017/01/15.
 */
public class LabeledFeatureVector extends FeatureVector implements Serializable {
  private final int y;

  protected double[] quadraticGradientBuffer;
  protected double[] linearGradientBuffer;
  protected int latentDimension;

  public LabeledFeatureVector(String format) {
    super(format);
    String[] ss = format.split(Constants.SPACE);
    this.y = Integer.parseInt(ss[0]);
  }

  public LabeledFeatureVector(int y, SortedSet<Feature> features) {
    super(features);
    this.y = y;
  }

  public int y() {
        return this.y;
    }

  public String toString() {
        return String.format("%d %s", this.y, super.toString());
    }

  public void initGradientBuffer(int latentDimension) {
    this.latentDimension = latentDimension;
    // linear gradient buffer
    if (this.linearGradientBuffer == null || this.linearGradientBuffer.length != this.size()) {
      this.linearGradientBuffer = new double[this.size()];
    }
    for (int i = 0; i < this.size(); ++i) {
      this.linearGradientBuffer[i] = 0.0;
    }

    // quadratic gradient buffer
    int l = this.size() * latentDimension;
    if (this.quadraticGradientBuffer == null || this.quadraticGradientBuffer.length != l) {
      this.quadraticGradientBuffer = new double[l];
    }
    for (int i = 0; i < l;++i) {
      this.quadraticGradientBuffer[i] = 0.0;
    }
  }

  public void updateQuadracticGradient(int idx, int k, double value) {
    try {
      this.quadraticGradientBuffer[k + idx * latentDimension] += value;
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public double getQudraticGradient(int idx, int k) {
    return this.quadraticGradientBuffer[k + idx * latentDimension];
  }

  public void updateLinearGradient(int idx, double value) {
        this.linearGradientBuffer[idx] += value;
    }

  public double getLinearGradient(int idx) {
    return this.linearGradientBuffer[idx];
  }

  @Override
  public int compareTo(FeatureVector o) {
    LabeledFeatureVector lfv = (LabeledFeatureVector)o;
    if (this.y == lfv.y()) {
      return super.compareTo(o);
    } else if (this.y > lfv.y()) {
      return -1;
    } else {
      return 1;
    }
  }
}
