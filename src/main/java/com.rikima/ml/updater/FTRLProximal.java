package com.rikima.ml.updater;

import com.rikima.ml.Updater;

/**
 * Created by a14350 on 2017/01/02.
 */
public class FTRLProximal implements Updater {
    private float epsilon = 1.0e-5f;
    private double alpha;
    private double beta;
    private double lambda1;

    private float[] curricurumSquaredGradients;
    private float[] z;

    public FTRLProximal(int size, double alpha, double beta, double lambda1) {
        this.alpha = alpha;
        this.beta = beta;
        this.lambda1 = lambda1;

        this.curricurumSquaredGradients = new float[size];
        this.z = new float[size];
    }

    public double getUpdate(int i, double gradient) {
        this.saveSquaredGradient(i, gradient);
        return this.alpha * gradient / Math.sqrt(this.curricurumSquaredGradients[i]+epsilon);
    }

    private void saveSquaredGradient(int index, double gradient) {
        this.curricurumSquaredGradients[index] += gradient * gradient;
    }

    private double sigma(int i, double gradient) {
        return ( Math.sqrt(this.curricurumSquaredGradients[i] + gradient * gradient) - Math.sqrt(this.curricurumSquaredGradients[i]) ) / this.alpha;
    }

    private double z(int i, double gradient) {
        double newZ = z[i] + gradient - this.sigma(i, gradient) * w(i, gradient);
        this.z[i] = (float)newZ;
        return newZ;
    }

    private double getEta(int i, double gradient) {
        return (Math.sqrt(this.curricurumSquaredGradients[i]) + this.beta) / this.alpha;
    }

    private double w(int i, double gradient) {
        double w = - this.getEta(i, gradient) * (z[i] - sign(z[i]) * this.lambda1);
        return w;
    }

    private int sign(float value) {
        return (value > 0) ? 1 : -1;
    }
}
