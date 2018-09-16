package com.rikima.ml;

import java.io.Serializable;

/**
 * Created by mrikitoku on 15/10/26.
 */
public class FieldFeature implements Comparable<FieldFeature>, Serializable {

  private int fieldId;
  private int featureId;
  private float value;

  public FieldFeature(int fieldId, int featureId, float value) {
    this.fieldId = fieldId;
    this.featureId = featureId;
    this.value = value;
  }

  public int getFieldId() {
        return this.fieldId;
    }

  public int getFeatureId() {
        return this.featureId;
    }

  public float getValue() {
        return this.value;
    }

  public int compareTo(FieldFeature o) {
    if (this.fieldId > o.getFieldId()) {
      return 1;
    } else if (this.fieldId < o.getFieldId()) {
      return -1;
    } else {
      return Integer.compare(this.featureId, o.getFeatureId());
    }
  }
}
