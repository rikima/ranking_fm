package com.rikima.ml;

import java.io.Serializable;
import java.util.Set;
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

  public void updateLinearGradient(int idx, double value) {
    this.linearGradientBuffer[idx] += value;
  }

  public double getLinearGradient(int idx) {
    return this.linearGradientBuffer[idx];
  }

  public void updateLinearGradients(double lambda,double linearRegParam,
                                    float[] linearWeights, Set<Integer> indexBuffer) {
    for (int i = 0; i < this.size(); ++i) {
      int idx_i = this.findex(i);
      double x_i = this.value(i);

      double g = lambda * x_i;
      if (!indexBuffer.contains(idx_i)) {
        g += linearRegParam * linearWeights[idx_i];
      }
      this.updateLinearGradient(i, g);
    }
  }

  public void updateLinearWeights(float[] linearWeights, Updater linearUpdater) {
    for (int i = 0; i < this.size(); ++i) {
      int idx_i = this.findex(i);
      double g = this.getLinearGradient(i);
      double update = linearUpdater.getUpdate(idx_i, g);
      linearWeights[idx_i] -= update;
    }
  }

  public void updateQuadracticGradient(int idx, int k, double value) {
    this.quadraticGradientBuffer[k + idx * latentDimension] += value;
  }

  public double getQudraticGradient(int idx, int k) {
    return this.quadraticGradientBuffer[k + idx * latentDimension];
  }

  public void updateQuadraticWeights(float[] quadraticWeights, Updater quadraticUpdater) {
    for (int i = 0; i < this.size(); ++i) {
      int idx_i = this.findex(i);
      int start = idx_i * latentDimension;
      for (int k = 0; k < this.latentDimension; ++k) {
        double g = this.getQudraticGradient(i, k);
        double update = quadraticUpdater.getUpdate(start+k, g);
        quadraticWeights[start+k] -= update;
      }
    }
  }

  public void updateQuadraticGradients(double lambda, double quadraticRegParam, float[] quadraticWeights,
                                       double[] buffer, Set<Integer> indexBuffer) {
    for (int i = 0; i < buffer.length; ++i) {
      buffer[i] = 0.0;
    }

    for (int i = 0; i < this.size(); ++i) {
      int idx_i = this.findex(i);
      double x_i = this.value(i);

      int start = idx_i * latentDimension;
      for (int k = 0; k < this.latentDimension; ++k) {
        buffer[k] += x_i * quadraticWeights[start + k];
      }
    }

    for (int i = 0; i < this.size(); ++i) {
      int idx_i = this.findex(i);
      double x_i = this.value(i);

      int start = idx_i * latentDimension;
      for (int k = 0; k < this.latentDimension; ++k) {
        double g_x = lambda * (x_i * buffer[k] - x_i * x_i * quadraticWeights[start + k]);
        double g = g_x;
        if (!indexBuffer.contains(idx_i)) {
          double g_r = quadraticRegParam * quadraticWeights[start + k];
          g += g_r;
        }
        this.updateQuadracticGradient(i, k, g);
      }
    }
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
