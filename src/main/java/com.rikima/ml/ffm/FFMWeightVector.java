package com.rikima.ml.ffm;

import com.rikima.ml.FieldFeatureVector;

import java.util.Random;

public class FFMWeightVector {

    private final int fieldDimension;
    private final int featureDimension;
    private final int latentDimension;

    private final float[] weights;

    public FFMWeightVector(int fieldDimension, int featureDimension, int latentDimension, float[] weights) {
        this.fieldDimension = fieldDimension;
        this.featureDimension = featureDimension;
        this.latentDimension = latentDimension;

        this.weights = weights;
        this.randomInit(1);
    }

    public void reset() {
        this.randomInit(System.currentTimeMillis());
    }


    public int size() {
        return this.weights.length;
    }

    public void randomInit(long seed) {
        this.randomInit(seed, 1.0e-5);
    }

    public void randomInit(long seed, double coef) {
        Random rand = new Random(seed);
        for (int i = 0; i < this.weights.length; ++i) {
            this.weights[i] = (float) coef * (rand.nextFloat() - 0.5f);
        }
    }

    public float wtx(FieldFeatureVector ffv) {
        float score = 0;
        for (int i = 0; i < ffv.size(); ++i) {
            int i1 = ffv.field(i) - 1;
            int j1 = ffv.feature(i) - 1;
            double v1 = ffv.value(i);
            for (int j = i + 1; j < ffv.size(); ++j) {

                int i2 = ffv.field(j) - 1;
                int j2 = ffv.feature(j) - 1;
                double v2 = ffv.value(j);

                int start1 = startIndex(i2, j1);
                int start2 = startIndex(i1, j2);
                for (int k = 0; k < latentDimension; ++k) {
                    score += v1 * v2 * this.weights[start1 + k] * this.weights[start2 + k];
                }
            }
        }
        return score;
    }


    // private methods --------------
    private int startIndex(int fieldIndex, int featureIndex) {
        return featureIndex * featureDimension * latentDimension + fieldIndex * latentDimension;
    }
}
