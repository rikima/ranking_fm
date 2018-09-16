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
    LabeledFeatureVector lfv = null;

    // bias term
    this.bias += 0;//coef; // TODO

    // linear term
    /* upper */
    {
      lfv = upper;
      for (int i = 0; i < lfv.size(); ++i) {
        int idx_i = lfv.findex(i);
        double x_i = lfv.value(i);

        double g = lambda * x_i + linearRegParam * this.linearWeights[idx_i];
        this.indexBuffer.add(idx_i);
        lfv.updateLinearGradient(i, g);
      }
    }
    /* lower */
    {
      lfv = lower;
      for (int i = 0; i < lfv.size(); ++i) {
        int idx_i = lfv.findex(i);
        double x_i = lfv.value(i);
        double g = -lambda * x_i;
        if (!this.indexBuffer.contains(idx_i)) {
          g += linearRegParam * this.linearWeights[idx_i];
        }
        lfv.updateLinearGradient(i, g);
      }
    }

    // quadratic term
    /* upper */
    {
      lfv = upper;
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
          double g_r = quadraticRegParam * this.quadraticWeights[start + k];
          double g_x = lambda * (x_i * buffer[k] - x_i * x_i * this.quadraticWeights[start + k]);
          double g = g_x + g_r;

          lfv.updateQuadracticGradient(i, k, g);
        }
      }
    }
    /* lower */
    {
      lfv = lower;
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
          double g_x = -lambda * (x_i * buffer[k] - x_i * x_i * this.quadraticWeights[start + k]);
          double g = g_x;
          if (!this.indexBuffer.contains(idx_i)) {
            double g_r = quadraticRegParam * this.quadraticWeights[start + k];
            g += g_r;
          }

          lfv.updateQuadracticGradient(i, k, g);
        }
      }
    }

    // update via adagrad
    // linear
    {
      lfv = upper;
      for (int i = 0; i < lfv.size(); ++i) {
        int idx_i = lfv.findex(i);
        double g = lfv.getLinearGradient(i);
        double update = linearUpdater.getUpdate(idx_i, g);
        this.linearWeights[idx_i] -= update;
      }

      lfv = lower;
      for (int i = 0; i < lfv.size(); ++i) {
        int idx_i = lfv.findex(i);
        double g = lfv.getLinearGradient(i);
        double update = linearUpdater.getUpdate(idx_i, g);
        this.linearWeights[idx_i] -= update;
      }
    }

    // quadratic
    {
      lfv = upper;
      for (int i = 0; i < lfv.size(); ++i) {
        int idx_i = lfv.findex(i);
        int start = this.startIndex(idx_i);
        for (int k = 0; k < this.latentDimension; ++k) {
          double g = lfv.getQudraticGradient(i, k);
          double update = quadraticUpdater.getUpdate(start+k, g);
          this.quadraticWeights[start+k] -= update;
        }
      }

      lfv = lower;
      for (int i = 0; i < lfv.size(); ++i) {
        int idx_i = lfv.findex(i);
        int start = this.startIndex(idx_i);
        for (int k = 0; k < this.latentDimension; ++k) {
          double g = lfv.getQudraticGradient(i, k);
          double update = quadraticUpdater.getUpdate(start+k, g);
          this.quadraticWeights[start+k] -= update;
        }
      }
    }
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
