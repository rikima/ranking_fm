package com.rikima.ml.fm;

import com.rikima.ml.updater.AdaGrad;
import com.rikima.ml.FeatureVector;
import com.rikima.ml.LabeledFeatureVector;
import com.rikima.ml.Updater;


/**
 * Created by rikitoku on 2017/01/02.
 */
public class FMTrainer {
  private FMWeightVector weightVector;

  private Updater linearUpdater;
  private Updater quadraticUpdater;

  private double alpha = 0.1;

  private final double linearRegParam;
  private final double quadraticRegParam;

  public FMTrainer(double lambda) {
    this.linearRegParam = lambda;
    this.quadraticRegParam = lambda;
  }

  public FMTrainer(double linearRegParam, double quadraticRegParam) {
    this.linearRegParam = linearRegParam;
    this.quadraticRegParam = quadraticRegParam;
  }

  public void init(int numLatentClasses, int featureDimension) {
    this.weightVector = new FMWeightVector(numLatentClasses, featureDimension);
    this.linearUpdater = new AdaGrad(this.weightVector.size(), this.alpha);
    this.quadraticUpdater = new AdaGrad(this.weightVector.size(), this.alpha);
  }


  public double train(LabeledFeatureVector lfv) {
    double score = this.score(lfv);
    double marginProb = this.marginProb(score, lfv.y());
    this.weightVector.update(lfv, marginProb, this.linearRegParam, this.quadraticRegParam, this.linearUpdater, this.quadraticUpdater);
    return this.logloss(score, lfv.y());
  }


  public double logloss(double score, int y) {
        return 1.0 * Math.log(1.0 + Math.exp(-y * score));
    }


  public double logloss(LabeledFeatureVector lfv) {
    double score = this.score(lfv);
    return this.logloss(score, lfv.y());
  }


  public double prob(FeatureVector fv) {
    double score = this.score(fv);
    return this.prob(score);
  }


  public double prob(double score) {
    double p = 1.0 / (1 + Math.exp(-score));
    return p;
  }


  public double score(FeatureVector fv) {
    double score = this.weightVector.wtx(fv);
    return score;
  }


  public double marginProb(double score, int y) {
    double p = 1.0 / (1 + Math.exp(-y * score));
    return p;
  }


  public double marginProb(LabeledFeatureVector lfv) {
    double score = this.score(lfv);
    return this.marginProb(score, lfv.y());
  }

}


// unused
