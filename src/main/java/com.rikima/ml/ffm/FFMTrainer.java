package com.rikima.ml.ffm;

import com.rikima.ml.LabeledFieldFeatureVector;
import com.rikima.ml.Updater;
import com.rikima.ml.FieldFeatureVector;

/**
 * Created by a14350 on 2017/01/02.
 */
public class FFMTrainer {
    private FFMWeightVector weightVector;
    private Updater updater;

    private double alpha = 0.1;
    private double lambda = 1.0;

    public FFMTrainer(double labmda) {
        this.lambda = labmda;
    }


    public void init(int numLatentClasses, int fieldDimension, int featureDimension) {
        //this.weightVector = new FFMWeightVector(numLatentClasses, fieldDimension, featureDimension);
        //this.updater = new AdaGrad(this.weightVector.size(), this.alpha);
    }


    public double train(LabeledFieldFeatureVector lffv) {
        /*
        double score = this.score(lffv);
        double marginProb = this.marginProb(score, lffv.y());
        this.weightVector.update(lffv, marginProb, lambda, this.updater);
        return this.logloss(score, lffv.y());
    */
        // TODO
        return 0.0;
    }


    public double logloss(double score, int y) {
        return 1.0 * Math.log(1.0 + Math.exp(-y * score));
    }


    public double logloss(LabeledFieldFeatureVector lffv) {
        //double score = this.score(lffv);
        //return this.logloss(score, lffv.y());
        // TODO
        return 0.0;
    }


    public double prob(FieldFeatureVector ffv) {
        double score = this.score(ffv);
        return this.prob(score);
    }


    public double prob(double score) {
        double p = 1.0 / (1 + Math.exp(-score));
        return p;
    }


    public double score(FieldFeatureVector ffv) {
        double score = this.weightVector.wtx(ffv);
        return score;
    }


    public double marginProb(double score, int y) {
        double p = 1.0 / (1 + Math.exp(-y * score));
        return p;
    }


    public double marginProb(LabeledFieldFeatureVector lffv) {
        /*
        double score = this.score(lffv);
        return this.marginProb(score, lffv.y());
    */
        // TODO
        return 0.0;
    }

}
