package deepDriver.dl.aml.math.test;

import deepDriver.dl.aml.math.JCudaBlasMathFunction;

public class TestJcuBlas {
	
	public static void main(String[] args) {
		float [][] a = new float[][]{
			{1,1,1},
			{1,1,1}
		};
		float [][] b = new float[][]{
			{1,1},
			{1,1},
			{1,1}
		};
		JCudaBlasMathFunction jbmf = new JCudaBlasMathFunction();
		float [][] c = new float[a.length][];
		for (int i = 0; i < c.length; i++) {
			c[i] = new float[b[0].length];
		}
		jbmf.multiple(a, b, c);
		syso(c);
	}
	
	public static void syso(float [][] a) {
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				System.out.print(a[i][j]+",");
			}
			System.out.println();
		}
	}

}
