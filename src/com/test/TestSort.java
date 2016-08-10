package com.test;

public class TestSort {
	
	public static void main(String[] args) {
		int [] aa = {20, 30, 4, 9, 5, 6, 99};
		
		System.out.println("before sorting...");
		for (int i = 0; i < aa.length; i++) {
			System.out.println(aa[i]);
		}
		
		for (int i = 0; i < aa.length; i++) {
			for (int j = i; j < aa.length; j++) {
				if (aa[i] > aa[j]) {
					int t = aa[i];
					aa[i] = aa[j];
					aa[j] = t;
				}
			}
		}
		System.out.println("after sorting...");
		for (int i = 0; i < aa.length; i++) {
			System.out.println(aa[i]);
		}
	}

}
