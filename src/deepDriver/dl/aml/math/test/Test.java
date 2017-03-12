package deepDriver.dl.aml.math.test;

import deepDriver.dl.aml.math.MathUtil;

public class Test {
	
	public static void main(String[] args) {
		double b = -2;
		double a = Math.sqrt(b);
		System.out.println(a);
		if (MathUtil.isNaN(a)) {
			System.out.println(" a it is "+a);
		} 
		if (MathUtil.isNaN(b)) {
			System.out.println("b it is "+b);
		} 
	}

}
