package deepDriver.dl.aml.math.test;

import deepDriver.dl.aml.math.LinearExp;

public class TestIExp4Function {
	
	public static void testCompute() {
		LinearExp le = new LinearExp(5);
		double [] x = new double[5];
		for (int i = 0; i < x.length; i++) {
			x[i] = 1.0;
		}
		
		le.compute(x);
		double r = le.getR();
		double [] v = le.getParameters();
		System.out.println("r: "+r);
		print(v);
		
	}
	
	public static void print(double [] v) {
		for (int i = 0; i < v.length; i++) {
			System.out.println(v[i]);
		}
	}
	
	public static void testDifCompute() {
		LinearExp le = new LinearExp(5);
		double [] x = new double[5];
		for (int i = 0; i < x.length; i++) {
			x[i] = 1.0;
		}
		le.difCompute(1.0, x);
		le.difCompute(1.0, x);
//		le.update(0.1, 0.1);
		System.out.println("p:");
		print(le.getParameters());
		System.out.println("dv:");
		print(le.getDv());
		System.out.println("dp:");
		print(le.getDeltaPara());
		le.update(0.1, 0.1);
		System.out.println("p:");
		print(le.getParameters());
	}
	
	public static void main(String[] args) {
//		TestLineExp.testCompute();
		testDifCompute();
	}

}
