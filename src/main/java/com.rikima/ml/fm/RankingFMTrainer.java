package com.rikima.ml.fm;

import com.rikima.ml.Updater;
import com.rikima.ml.FeatureVector;
import com.rikima.ml.LabeledFeatureVector;
import com.rikima.ml.updater.AdaGrad;

public class RankingFMTrainer {
    private FMWeightVector weightVector;

    private Updater linearUpdater;
    private Updater quadraticUpdater;

    private double alpha = 0.1;

    private final double linearRegParam;
    private final double quadraticRegParam;

    public RankingFMTrainer(double lambda) {
        this.linearRegParam = lambda;
        this.quadraticRegParam = lambda;
    }

    public RankingFMTrainer(double linearRegParam, double quadraticRegParam) {
        this.linearRegParam = linearRegParam;
        this.quadraticRegParam = quadraticRegParam;
    }

    public void init(int numLatentClasses, int featureDimension) {
        this.weightVector = new FMWeightVector(numLatentClasses, featureDimension);
        this.linearUpdater = new AdaGrad(this.weightVector.size(), this.alpha);
        this.quadraticUpdater = new AdaGrad(this.weightVector.size(), this.alpha);
    }


  public double train(LabeledFeatureVector upper, LabeledFeatureVector lower) {
    double lambda = calcLambda(upper, lower);
    this.weightVector.updateForRanking(upper, lower, lambda, this.linearRegParam, this.quadraticRegParam,
            this.linearUpdater, this.quadraticUpdater);
    return logloss(upper, lower);
  }

  public double logloss(LabeledFeatureVector upper, LabeledFeatureVector lower) {
    double upperScore = score(upper);
    double lowerScore = score(lower);
    double loss =  Math.log(1.0 + Math.exp(- (upperScore - lowerScore) ) );
    return loss;
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

  private double calcLambda(FeatureVector upper, FeatureVector lower) {
    double upperScore = score(upper);
    double lowerScore = score(lower);

    double lambda = -1.0 / (1.0 + Math.exp(upperScore - lowerScore) );
    return lambda;
  }

}
