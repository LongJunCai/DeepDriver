package deepDriver.dl.aml.math.test;

import deepDriver.dl.aml.math.SoftMaxExp;

public class TestSoftMaxExp {
	
	public static void testCompute() {
		SoftMaxExp sme = new SoftMaxExp(3, 5, 1.0);
		double [] x = new double[5];
		for (int i = 0; i < x.length; i++) {
			x[i] = 1.0;
		}
		sme.compute(x); 
	}
	
	public static void testDifCompute() {
		SoftMaxExp sme = new SoftMaxExp(3, 5, 1.0);
		double [] x = new double[5];
		for (int i = 0; i < x.length; i++) {
			x[i] = 1.0;
		}
		double [] dy = new double[]{1.0, 1.0, 1.0};
		sme.difCompute(dy, x);
	}
	
	public static void main(String[] args) {
//		testCompute();
		testDifCompute();
	}

}
