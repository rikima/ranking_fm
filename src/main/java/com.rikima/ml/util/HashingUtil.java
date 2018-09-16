package com.rikima.ml.util;

import org.apache.mahout.math.MurmurHash;

import java.io.Serializable;

/**
 * Created by a14350 on 2016/06/21.
 */
public class HashingUtil implements Serializable {
    public static final int NUM_FEATURES = 1 << 24;

    public static byte[] bytesForString(String x) {
        if (x == null) {
            return new byte[0];
        } else {
            return x.getBytes();
        }
    }

    //public static int hash(String feature, int seed) {
    //return hash(feature, NUM_FEATURES, seed);
    //}

    public static int hash(String feature, int numFeatures, int seed) {
        long r =  MurmurHash.hash64A(bytesForString(feature), seed) % numFeatures;
        if (r < 0) r += numFeatures;
        return (int)r;
    }

}
