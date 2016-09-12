package deepDriver.dl.aml.math;

public class MathUtil {
	
	public static double [] scale(double [] v1, double v1p) { 
		for (int i = 0; i < v1.length; i++) {
			v1[i] = v1p * v1[i]; 
		}
		return v1;
	}
	
	public static double [] plus2V(double [] v1, double v1p, double [] v2) { 
		for (int i = 0; i < v2.length; i++) {
			v2[i] = v1p * v1[i] +  v2[i]; 
		}
		return v2;
	}
	
	public static double [] plus2V(double [] v1, double [] v2) { 
		return plus2V(v1, 1.0, v2); 
	}
	
	public static double multiple(double [] v1, double [] v2) { 
		double v = 0;
		for (int i = 0; i < v1.length; i++) {
			v = v + v1[i] * v2[i]; 
		}
		return v;
	}
	
	public static double cos(double [] v1, double [] v2) {
		double s = 0;
		s = multiple(v1, v2);
		double a = Math.pow(multiple(v1, v1) * multiple(v2, v2), 0.5);
		return s/a;
	}

}
