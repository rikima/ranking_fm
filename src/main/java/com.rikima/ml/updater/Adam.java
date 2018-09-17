package com.rikima.ml.updater;

import com.rikima.ml.Updater;

/**
 * Created by a14350 on 2017/01/02.
 */
class Adam implements Updater {
    public static final double alpha   = 0.001;
    public static final double beta1   = 0.9;
    public static final double beta2   = 0.999;
    public static final double epsilon = 1.0e-8;

    private float[] m;
    private float[] v;

    private int t = 1;

    public Adam(int size) {
        this.m = new float[size];
        this.v = new float[size];
    }

    @Override
    public double getUpdate(int i, double gradient) {
        this.m[i] *= beta1;
        this.m[i] = (float)(beta1 * this.m[i] + (1.0 - beta1) * gradient);
        double mt = this.m[i] / (1.0 - Math.pow(beta1, t));

        this.v[i] = (float)(beta2 * this.v[i] + (1.0 - beta2) * gradient * gradient);
        double vt = (this.v[i] / (1.0 - Math.pow(beta2, t)));

        t++;

        double update = alpha / (Math.sqrt(vt) + epsilon) * mt;
        return update;
    }
}
