package com.test;

public class TestArray {
	
	public static void main(String[] args) {
		double [] da = new double[60];
		for (int i = 0; i < da.length; i++) {
			da[i] = i;
		}
		
//		for (int i = 0; i < da.length; i++) {
//			System.out.println(da[i]);
//		}
		
		String [] s = new String[60];
		for (int i = 0; i < s.length; i++) {
			s[i] = i + "";
		}
		for (int i = 0; i < s.length; i++) {
			System.out.println(s[i]);
		}
	}

}
