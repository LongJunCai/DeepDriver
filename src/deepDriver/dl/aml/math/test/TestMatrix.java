package deepDriver.dl.aml.math.test;

import deepDriver.dl.aml.math.MathUtil;

public class TestMatrix {
	
	public static void testSum() {
		double [] a = {1,1,1,1,1};
		double s = MathUtil.sum(a);
		System.out.println(s);
	}
	
	public static void testDifMultipleY() {
		
		double [][] dr = {
				{1, 2},
				{3, 4}
		};
		
		double [][] x = {
				{1, 2},
				{3, 4}
		};
		double [][] dy = MathUtil.difMultipleY(dr, x);
		print(dy);
	}
	
	public static void testDifMultipleX() {
		
		double [][] dr = {
				{1, 2},
				{3, 4}
		};
		
		double [][] y = {
				{1, 2},
				{3, 4}
		};
		double [][] dx = MathUtil.difMultipleX(dr, y);
		print(dx);
	}
	
	public static void testMultipleV() {
		double [][] x = {
				{1,1,1},
				{2,2,2}
		};
		
		double [] y = {1,2,1} ;
		double [][] r = MathUtil.multipleV(x, y);
		print(r);
	}
	
	public static void print(double [][] r) {
		for (int i = 0; i < r.length; i++) {
			for (int j = 0; j < r[i].length; j++) {
				System.out.print(r[i][j]+"\t");
			}
			System.out.println();
		}
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
		print(r);
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
//		testDifMultipleX();
//		testDifMultipleY();
//		testMultipleV();
//		testSort();
		mt();
	}
	
	
	public static void mt() {
		int l = 300;
		double [][] a = new double[l][];
		for (int i = 0; i < a.length; i++) {
			a[i] = new double[l];
			for (int j = 0; j < a[i].length; j++) {
				a[i][j] = i * j;
			}
		}
		
		double [][] b = new double[l][];
		for (int i = 0; i < b.length; i++) {
			b[i] = new double[l];
			for (int j = 0; j < b[i].length; j++) {
				b[i][j] = i * j;
			}
		}
		double t = System.currentTimeMillis();
		MathUtil.setThreadCnt(1);
		double [][] c = MathUtil.multiple(a, b);
		System.out.println(System.currentTimeMillis() - t);
		/**		
		Array2DRowRealMatrix m = new Array2DRowRealMatrix(a);
		Array2DRowRealMatrix m2 = m.multiply(new Array2DRowRealMatrix(b));
		
		*41983.0**/
	}

}
