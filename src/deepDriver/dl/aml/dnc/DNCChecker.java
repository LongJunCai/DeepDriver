package deepDriver.dl.aml.dnc;

import deepDriver.dl.aml.math.MathUtil;

public class DNCChecker {
	
	public static boolean checkBg(double [][] x, String s, int t) {
		if (!debug) {
			return false;
		}
		for (int i = 0; i < x.length; i++) {
			boolean b = checkBg(x[i], s, t);
			if (b) {				
				return true;
			}
		}
		return false;
	}
	
	static boolean bg = false;  
	public static boolean checkBg(double [] x, String s, int t) {
		if (!debug) {
			return false;
		}
		if (!bg) {			
			if (checkNormal(x, s, t)) {
				bg = true;
				return bg;
			} 
		}
		return false;
	}
	
	static boolean debug = false;
	
	public static boolean checkSimplex(double [][] x, String s, int t) {
		if (!debug) {
			return false;
		}
		for (int i = 0; i < x.length; i++) {
			if (checkSimplex(x[i], s, t)) {
				return true;
			}
		}
		return false;
	}
	
	public static boolean checkSimplex(double [] x, String s, int t) {
		if (!debug) {
			return false;
		}
		double st = MathUtil.checkNormal(x);
		if (st > 1.1) {
			System.out.println(s+"="+st+", not simplex, "+ t);
			return true;
		} 
		return false;
	}
	
	public static boolean checkNormal(double [] x, String name, int t) {
		if (!debug) {
			return false;
		}
		boolean b = MathUtil.isNaN(x);
		if (b) {
			System.out.println(name +" is NaN, "+t);
			print(x);
		}
		return b;
	}
	
	static boolean b = false;
	
	public static void print(double [][] x) {
		if (!debug) {
			return ;
		}
//		print(x[0]);
		for (int i = 0; i < x.length; i++) {
			print(x[i]);
		}
	}
	
	public static void print(double [] x) {
		if (!debug) {
			return ;
		}
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < x.length; i++) {
			sb.append(x[i]+",");
		}
		System.out.println(sb.toString());
	}
	
	public static void checkElementLess1(double [] x, String s, int t) {
		if (!debug) {
			return;
		}
		double sum = 0;
		for (int i = 0; i < x.length; i++) {
			if (x[i] > 1) {
				System.out.println(s+" is larger than 1, time is: "+t);
			}
		}		
	}

}
