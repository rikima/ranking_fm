package com.rikima.ml.eval;

import mloss.roc.Curve;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by mrikitoku on 15/08/31.
 */
public class Evaluation {
  public static class ResultItem implements Comparable<ResultItem> {
    double score;
    int expected;
    int actual;

    ResultItem(int expected, int actual, double score) {
      this.expected = expected;
      this.actual = actual;
      this.score = score;
    }

    @Override
    public int compareTo(ResultItem o) {
      if (this.score < o.score) {
        return 1;
      } else if (this.score > o.score) {
        return -1;
      } else {
        return 0;
      }
    }

    public String toTSV() {
            return String.format("%d\t%d\t%f", this.expected, this.actual, this.score);
        }
  }

  private List<Integer> expected;
  private List<Integer> actual;
  private List<Double> scores;

  private List<ResultItem> sortedResults = new ArrayList<ResultItem>();

  private int pp = 0;
  private int pn = 0;
  private int np = 0;
  private int nn = 0;

  // methods ------------
  public Integer[] getExpecteds() {
    Integer[] array = new Integer[this.expected.size()];
    this.expected.toArray(array);
    return array;
  }

  public Integer[] getActuals() {
    Integer[] array = new Integer[this.actual.size()];
    this.actual.toArray(array);

    return array;
  }

  public Double[] getScores() {
    Double[] scores = new Double[this.scores.size()];
    this.scores.toArray(scores);

    return scores;
  }

  public List<ResultItem> getScoredResult() {
        return this.sortedResults;
    }

  public int pp() {
        return pp;
    }

  public int pn() {
        return pn;
    }

  public int np() {
        return np;
    }

  public int nn() {
        return nn;
    }

  public int numPositives() {
        return (pp + pn);
    }
  public int numPositiveOutputs() {
        return (pp + np);
    }

  public int numNegatives() {
        return (nn + nn);
    }

  public int numTotal() {
        return (pp + pn + np + nn);
    }

  public Evaluation() {
    this.expected = new ArrayList<Integer>();
    this.actual   = new ArrayList<Integer>();
    this.scores   = new ArrayList<Double>();
  }

  public void setResult(int expectedY, int actualY) {
    this.expected.add(expectedY);
    this.actual.add(actualY);
  }

  public void setResult(int expectedY, int actualY, double score) {
    this.expected.add(expectedY);
    this.actual.add(actualY);
    //double prob = 1.0 / (1 + Math.exp(-10 * score));
    //this.scores.add(prob);
    this.scores.add(score);
    this.sortedResults.add(new ResultItem(expectedY, actualY, score));
  }

  public double averagePrecision() {
    Collections.sort(this.sortedResults);
    assert this.sortedResults.size() == this.expected.size();
    assert this.sortedResults.size() == this.actual.size();
    assert this.sortedResults.size() == this.scores.size();

    double ap = 0.0;
    int pp = 0;
    int pn = 0;
    int c = 0;
    for (ResultItem ri : this.sortedResults) {
      c++;
      if (ri.expected > 0) {
        if (ri.expected == ri.actual) pp += 1;
        else pn += 1;
        double precision = (double) pp / c;
        ap += precision;
      }
    }
    if (pn + pp > 0) {
      return ap / (pp + pn);
    }
    return 0.0;
  }

  public double auc() {
    Curve analysis = new Curve.PrimitivesBuilder()
                             .scores(this.scores)
                             .labels(this.expected)
                             .build();
    // Calculate AUC ROC
    double area = analysis.rocArea();
    /*
        for (int i = 0; i < this.actual.size(); ++i) {
            System.out.println(String.format("%f\t%d", this.scores.get(i), this.actual.get(i)));
        }
        */
    return area;
  }

  public void printResult(String comment) {
        this.printResult(comment, false);
    }

  public void printResult(String comment, boolean verbose) {
    pp = 0;
    pn = 0;
    np = 0;
    nn = 0;
    int size = this.expected.size();
    for (int i = 0; i < size; ++i) {
      assert this.expected != null;
      int ey = this.expected.get(i);

      assert this.actual != null;
      int ay = this.actual.get(i);

      if (ey * ay > 0) {
        if (ey > 0) {
          pp++;
        } else {
          nn++;
        }
      } else {
        if (ey > 0) {
          pn++;
        } else {
          np++;
        }
      }
    }

    if (verbose) {
      for (int i = 0; i < size; ++i) {
        int ey = this.expected.get(i);
        int ay = this.actual.get(i);
        double score = this.scores.get(i);
        System.out.println(String.format("%f\t%d\t%d", score, ey, ay));
      }
    }

    System.out.println("\n### " + comment);
    System.out.println(String.format("PP: %d PN: %d NP: %d NN: %d", this.pp, this.pn, this.np, this.nn));
    {//accuracy
      double total = this.pp + this.pn + this.np + this.nn;
      double acc = (this.pp + this.nn) / total;
      System.out.println(String.format("Accuracy: %f", acc));
      System.out.println(String.format("Error rate: %f", 1.0 - acc));
    }
    System.out.println("---");
    {// positive perf
      double recall = (pp) / (double) (pp + pn);
      double prec = (pp) / (double) (pp + np);
      double f1 = 2.0 * recall * prec / (recall + prec);

      System.out.println(String.format("positive Prec: %f", prec));
      System.out.println(String.format("positive Recall: %f", recall));
      System.out.println(String.format("positive F1: %f", f1));
    }
    System.out.println("---");
    {// negative perf
      double recall = (nn) / (double) (nn + pn);
      double prec = (nn) / (double) (nn + np);
      double f1 = 2.0 * recall * prec / (recall + prec);

      System.out.println(String.format("negative Prec: %f", prec));
      System.out.println(String.format("negative Recall: %f", recall));
      System.out.println(String.format("negative F1: %f", f1));
    }
    System.out.println("---");
    {// aucroc score
      System.out.println(String.format("AUC: %f", this.auc()));
    }
    System.out.println("---");
    {// ap score
      System.out.println(String.format("Avg. Prec: %f", this.averagePrecision()));
    }

  }
}
