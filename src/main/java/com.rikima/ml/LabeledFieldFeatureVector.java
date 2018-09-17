package com.rikima.ml;

import java.io.Serializable;

/**
 * Created by a14350 on 2017/01/04.
 */
public class LabeledFieldFeatureVector extends FieldFeatureVector implements Serializable {
    private int y = 0;

    protected LabeledFieldFeatureVector(int size) {
        super(size);
    }

    public void setY(int y) {
        this.y = y;
    }

    public int y() {
        return this.y;
    }

}
