package com.rikima.ml;

public class Feature implements Comparable<Feature> {
    protected int id;
    protected double val;

    public Feature(int fid, double val2) {
        this.id = fid;
        this.val = val2;
    }

    public int id() {
        assert this.id > 0;
        return this.id;
    }

    public int index() {
        return this.id()-1;
    }

    public double val() {
    	return this.val;
    }
    
    public void setVal(double val) {
    	this.val = (float)val;
    }

    public int compareTo(Feature o) {
        if (this.id == o.id()) return 0;
        else if (this.id < o.id()) return -1;
        else return 1;
    }

    public String toString() {
        return String.format("%d:%f", this.id, this.val);
    }

}
