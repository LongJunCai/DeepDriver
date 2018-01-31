package deepDriver.dl.aml.math.test;

import org.jblas.FloatMatrix;

import deepDriver.dl.aml.math.BlasMathFunction;

public class TestMathUtils {
	
	public static void main(String[] args) {
		float [][] a = new float[][]{
			{1,1,1},
			{1,1,1}
		};
		float [][] b = new float[][]{
			{1,1,1},
			{1,1,1}
		};
		float [][] c = new float[][]{
			{1,1,1},
			{1,1,1}
		};
		BlasMathFunction bmf = new BlasMathFunction();
		bmf.plus(a, 1.2f, b, 2.3f, c);
//		System.out.println(bmf);
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
	
	public static void main2(String[] args) {
		float [][] a = new float[][]{
			{1,1,1},
			{1,1,1}
		};
		FloatMatrix fm = new FloatMatrix(a);
		fm.addi(1.2f);
		System.out.println(fm);
	}

}
