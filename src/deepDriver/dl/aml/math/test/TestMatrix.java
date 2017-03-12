package deepDriver.dl.aml.math.test;

import deepDriver.dl.aml.math.MathUtil;

public class TestMatrix {
	
	public static void testSum() {
		double [] a = {1,1,1,1,1};
		double s = MathUtil.sum(a);
		System.out.println(s);
	}
	
	public static void testMultiple() {
		double [][] x = {
				{1,1,1},
				{2,2,2}
		};
		
		double [][] y = {
				{1,2},
				{1,2},
				{1,2}
		};
		double [][] r = MathUtil.multiple(x, y);
		for (int i = 0; i < r.length; i++) {
			for (int j = 0; j < r[i].length; j++) {
				System.out.println(r[i][j]);
			}
		}
	}
	
	public static void testSort() { 
		double [] ws = {0.1, 0.2, 0.3, 0.4, 0.0001, 0.00002, 0.00003, 0.00004, 0.0005};
		int k = 3;
		double s = MathUtil.sumMaxK(ws, k);
		MathUtil.simplex2(ws, s, k);
		for (int i = 0; i < ws.length; i++) {
			System.out.print(ws[i]+",");
		}
	}
	
	public static void main(String[] args) {
//		testSum();
//		testMultiple();
		testSort();
	}

}
