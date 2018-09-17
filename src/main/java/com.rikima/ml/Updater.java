package com.rikima.ml;

/**
 * Created by a14350 on 2017/01/07.
 */
public interface Updater {
  double getUpdate(int i, double gradient);
}
