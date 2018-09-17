package com.rikima.ml.updater;

import com.rikima.ml.Updater;

public class SGD implements Updater {

    @Override
    public double getUpdate(int i, double gradient) {
        return gradient;
    }
}
