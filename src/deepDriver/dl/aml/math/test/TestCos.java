package deepDriver.dl.aml.math.test;

import deepDriver.dl.aml.math.MathUtil;

public class TestCos {
	
	public static void testDifCos() {
		double [] v1 = {1, 1};
		double [] v2 = {-1, -2};
		double [] dv1 = new double[v1.length];
		System.out.println(MathUtil.cos(v1, v2));
		MathUtil.difCos(0.1, dv1, v1, v2);
		for (int i = 0; i < dv1.length; i++) {
			System.out.println(dv1[i]);
		}
	}
	
	public static void testCos() {
		double [] v1 = {1, 1};
		double [][] v2 = {
				{-1, -1},
				{0, 1},
				{1, 0},
				{0, -1},
				{0, -2},
				{1, -1},
				{-1, 1},
				{-1, 0},
				{-2, 0}
		};    
		for (int i = 0; i < v2.length; i++) {
			System.out.println("v2["+i+
					"] cos is "+ MathUtil.cos(v1, v2[i]));
		}
		
	}
	
	public static void testDifSoftmax() {
		double [] v1 = {1.0, 0, -1.0, 0.7, -0.7};
		double [] dr = {0.1, 0.2, 0.3, 0.4, 0.5};
		double [] sf = MathUtil.difSoftMax4Weighting(dr, v1, 1.5);
		for (int i = 0; i < sf.length; i++) {
			double d = sf[i];
			System.out.println(d);
		}
		double db = MathUtil.difSoftMax4Beta(dr, v1, 1.5);
		System.out.println("db is "+db);
	}
	
	public static void testSoftmax() {
		double [] v1 = {1.0, 0, -1.0, 0.7, -0.7};
		double [] sf = MathUtil.softMax(v1, 2);
		for (int i = 0; i < sf.length; i++) {
			double d = sf[i];
			System.out.println(d);
		}
	}
	
	public static void main(String[] args) {
//		testCos();
//		testSoftmax();
//		testDifSoftmax();
		testDifCos();
	}

}
