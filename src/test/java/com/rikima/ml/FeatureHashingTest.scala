package com.rikima.ml;

import org.junit.Assert._
import org.junit.Test

class FeatureHashingTest {
  val hashing = new FeatureHashing
  @Test
  def testGetFeatureHashing(): Unit = {
    val libsvmFormat1 = f" men:1.0 age_24:1.0 pref_23:1.9"
    val libsvmFormat2 = f" age_24:1.0 men:1.0 pref_23:1.9"
    println(hashing.getHashingSvmformat(libsvmFormat1, false))
    println(hashing.getHashingSvmformat(libsvmFormat2, false))
    assertEquals(hashing.getHashingSvmformat(libsvmFormat1,false), hashing.getHashingSvmformat(libsvmFormat2, false))

  }

}
