package com.rikima.ml.updater;

import com.rikima.ml.Updater;

/**
 * Created by a14350 on 2017/01/02.
 */
public class AdaGrad implements Updater {
    private float epsilon = 1.0e-5f;
    private double alpha;

    private float[] curricurumSquaredGradients;

    public AdaGrad(int size, double alpha) {
        this.alpha = alpha;
        this.curricurumSquaredGradients = new float[size];
        for (int i = 0; i < size; ++i) {
            this.curricurumSquaredGradients[i] = 1.0f;
        }
    }

    public double getUpdate(int i, double gradient) {
        this.saveSquaredGradient(i, gradient);
        return this.alpha * gradient / Math.sqrt(this.curricurumSquaredGradients[i]+epsilon);
    }

    private void saveSquaredGradient(int index, double gradient) {
        this.curricurumSquaredGradients[index] += gradient*gradient;
    }

}
