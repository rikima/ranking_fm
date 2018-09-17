package com.rikima.ml.eval;

import com.rikima.ml.LabeledFeatureVector;

import org.ltr4l.evaluation.DCG;
import org.ltr4l.query.Document;

import java.util.*;

/**
 * Created by mrikitoku on 15/08/31.
 */
public class RankingEvaluation {

  public static class RankingResult {
    TreeSet<RankingResultItem> result = new TreeSet<>(RankingResultItem::compareTo);

    public void addRankingResultItem(double score, LabeledFeatureVector lfv) {
      this.result.add(new RankingResultItem(score, lfv));
    }

    public List<Document> getRanking() {
      ArrayList<Document> rv = new ArrayList<Document>();
      for (RankingResultItem item : result) {
        Document doc = new Document();
        doc.setLabel(item.getLabel());
        rv.add(doc);
      }
      return rv;
    }
  }

  public static class RankingResultItem implements Comparable<RankingResultItem> {
    double score;
    LabeledFeatureVector lfv;

    public RankingResultItem(double score, LabeledFeatureVector lfv) {
      this.lfv = lfv;
      this.score = score;
    }

    public int getLabel() {
      return this.lfv.y();
    }

    @Override
    public int compareTo(RankingResultItem o) {
      if (this.score < o.score) {
        return 1;
      } else if (this.score > o.score) {
        return -1;
      } else {
        return 0;
      }
    }

    public String toTSV() {
      return String.format("%f\t%s", this.score, this.lfv.toString());
    }
  }

  Map<String, RankingResult> results = new HashMap<String, RankingResult>();

  DCG.NDCG ndcg = new DCG.NDCG();

  // methods ------------

  public void prepareResult(String query) {
    this.results.put(query, new RankingResult());
  }

  public void addResult(String query, double score, LabeledFeatureVector lfv) {
    this.results.get(query).addRankingResultItem(score, lfv);
  }


  public void printResult(String comment, int position) {
    this.printResult(comment, position, false);
  }

  public int size() {
    return this.results.size();
  }

  public double getTotalNDCG(int position) {
    TreeSet<String> queries = new TreeSet<String>();
    queries.addAll(this.results.keySet());
    double score = 0;
    for (String query : queries) {
      RankingResult result = this.results.get(query);
      score += ndcg.calculate(result.getRanking(), position);;
    }

    return score;
  }

  public void printResult(String comment,int position,  boolean verbose) {
    System.out.println("#" + comment);
    System.out.println("#queries:" + this.results.keySet().size());
    double ndcg = getTotalNDCG(position) / size();
    System.out.printf("%s avg. ndcg %f \n", comment, ndcg);
  }
}
