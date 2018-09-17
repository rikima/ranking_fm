package com.rikima.ml.fm;

import java.util.*;

import com.rikima.ml.FeatureVector;
import com.rikima.ml.LabeledFeatureVector;
import com.rikima.ml.Updater;

/**
 * Created by rikitoku on 2017/01/03.
 */
public class FMWeightVector {
  private final int featureDimension;
  private final int latentDimension;

  private float bias;
  private final float[] linearWeights;
  private final float[] quadraticWeights;

  private final double[] buffer;
  private final Set<Integer> indexBuffer = new HashSet<Integer>();

  /**
   * constructor
   *
   * @param numLatentClasses
   * @param numFeatures
   */
  public FMWeightVector(int numLatentClasses, int numFeatures) {
    this.featureDimension = numFeatures;
    this.latentDimension = numLatentClasses;
    this.bias = 0;
    this.linearWeights = new float[featureDimension];
    this.quadraticWeights = new float[featureDimension * latentDimension];
    this.randomInit(1);
    this.buffer = new double[latentDimension];
  }

  public double wtx(FeatureVector fv) {
    double score = bias;
    for (int i = 0; i < fv.size(); ++i) {
      int f_i = fv.findex(i);
      double x_i = fv.value(i);

      // linear term
      score += x_i * this.linearWeights[f_i];

      // quadratic term
      for (int j = i + 1; j < fv.size(); ++j) {
        int f_j = fv.findex(j);
        double x_j = fv.value(j);

        int start1 = f_i * latentDimension;
        int start2 = f_j * latentDimension;
        for (int k = 0; k < latentDimension; ++k) {
          score += x_i * x_j * this.quadraticWeights[start1+k] * this.quadraticWeights[start2+k];
        }
      }
    }

    return score;
  }


  public double wtx2(FeatureVector fv) {
    this.clearBuffer();
    for (int i = 0; i < fv.size(); ++i) {
      int fidx = fv.findex(i);
      double v = fv.value(i);

      int start = this.startIndex(fidx);
      for (int k = 0; k < latentDimension; ++k) {
        this.buffer[k] += v * this.quadraticWeights[start+k];
      }
    }

    double score = bias;
    for (int k = 0; k < latentDimension; ++k) {
      score += this.buffer[k] * this.buffer[k];
    }

    this.clearBuffer();
    for (int i = 0; i < fv.size(); ++i) {
      int fidx = fv.findex(i);
      double v = fv.value(i);

      int start = this.startIndex(fidx);

      // linear term
      score += v *  this.linearWeights[fidx];

      // quadratic term
      for (int k = 0; k < latentDimension; ++k) {
        this.buffer[k] += this.quadraticWeights[start + k] * this.quadraticWeights[start + k] * v * v;
      }
    }
    for (int k = 0; k < latentDimension; ++k) {
      score -= this.buffer[k] * this.buffer[k];
    }
    return 0.5 * score;
  }

  public int size() {
        return this.quadraticWeights.length;
    }

  /**
   * update for Ranking FM
   * @param upper
   * @param lower
   * @param lambda
   * @param linearRegParam
   * @param quadraticRegParam
   * @param linearUpdater
   * @param quadraticUpdater
   */
  public void updateForRanking(LabeledFeatureVector upper, LabeledFeatureVector lower, double lambda,
                               double linearRegParam, double quadraticRegParam,
                               Updater linearUpdater, Updater quadraticUpdater) {
    upper.initGradientBuffer(latentDimension);
    lower.initGradientBuffer(latentDimension);
    this.indexBuffer.clear();

    // bias term
    this.bias += 0;//coef; // TODO

    // linear term
    /* upper */
    upper.updateLinearGradients(lambda, linearRegParam, this.linearWeights, indexBuffer);
    lower.updateLinearGradients(-lambda, linearRegParam, this.linearWeights, indexBuffer);

    // quadratic term
    upper.updateQuadraticGradients(lambda, quadraticRegParam, this.quadraticWeights, buffer, indexBuffer);
    lower.updateQuadraticGradients(-lambda, quadraticRegParam, quadraticWeights, buffer, indexBuffer);

    // update via adagrad
    // linear
    upper.updateLinearWeights(this.linearWeights, linearUpdater);
    lower.updateLinearWeights(this.linearWeights, linearUpdater);

    // quadratic
    upper.updateQuadraticWeights(this.quadraticWeights, quadraticUpdater);
    lower.updateQuadraticWeights(this.quadraticWeights, quadraticUpdater);
  }

  /**
   * update for usual FM
   * @param lfv
   * @param marginProb
   * @param linearRegParam
   * @param quadraticRegParam
   * @param linearUpdater
   * @param quadraticUpdater
   */
  public void update(LabeledFeatureVector lfv, double marginProb, double linearRegParam, double quadraticRegParam,
                     Updater linearUpdater, Updater quadraticUpdater) {

    int y = lfv.y();
    double coef = y * (1.0 - marginProb);

    // bias term
    this.bias += 0  ;//coef; // TODO

    // linear term
    for (int i = 0; i < lfv.size(); ++i) {
      int findex = lfv.findex(i);
      double x_i = lfv.value(i);
      double g = coef * x_i + linearRegParam * this.linearWeights[findex];

      double update = linearUpdater.getUpdate(findex, g);
      this.linearWeights[findex] += update;
    }

    // quadratic term
    this.clearBuffer();
    for (int i = 0; i < lfv.size(); ++i) {
      int idx_i = lfv.findex(i);
      double x_i = lfv.value(i);

      int start = this.startIndex(idx_i);
      for (int k = 0; k < this.latentDimension; ++k) {
        buffer[k] += x_i * this.quadraticWeights[start + k];
      }
    }

    for (int i = 0; i < lfv.size(); ++i) {
      int idx_i = lfv.findex(i);
      double x_i = lfv.value(i);

      int start = this.startIndex(idx_i);
      for (int k = 0; k < this.latentDimension; ++k) {
        double g_x = coef * (x_i * buffer[k] - x_i * x_i * this.quadraticWeights[start+k]);
        double g_r = quadraticRegParam * this.quadraticWeights[start + k];
        double g = g_x + g_r;

        double update = quadraticUpdater.getUpdate(start+k, g);
        this.quadraticWeights[start+k] += update;
      }
    }
  }

  private void randomInit(long seed) {
    float coef = 0.01f;
    Random rand = new Random(seed);
    for (int i = 0; i < this.quadraticWeights.length; ++i) {
      this.quadraticWeights[i] = rand.nextFloat() * coef;
    }
    for (int i = 0; i < this.linearWeights.length; ++i) {
      this.linearWeights[i] = rand.nextFloat() * coef;
    }
  }

  private int startIndex(int featureIndex) {
    return featureIndex * latentDimension;
  }

  private void clearBuffer() {
    for (int k = 0; k < latentDimension; ++k) {
      this.buffer[k] = 0;
    }
  }
}
