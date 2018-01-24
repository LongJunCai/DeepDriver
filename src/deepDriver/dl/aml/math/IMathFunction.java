package deepDriver.dl.aml.math;

public interface IMathFunction {
	
	public int getThreadCnt();

	public void setThreadCnt(int threadCnt);
	
//	public static double sum(double [] v1);
//	
//	public static double sum(double [][] x, int ci);
//	
//	public static double [] plus(double [] v1, double v1p);
//	
//	public static double checkNormal(double [] x);
//	
//	public static double normalize(double [] x);
//	
//	public static double [][] allocateE(int r, int c);
//	
//	public static void initMatrix(double [][] x, double length, double min);
//	
//	public static int [][] allocateInt(int r, int c);
//	
//	public static void reset(boolean [] bs);
//	
//	
//	public static double [][] allocate(int r, int c);
//	
//	public static void reset2zero(double [] x);
//	
//	public static void reset2zero(double [][][] x);
//	
//	public static void reset2zero(double [][] x);
//	
//	public static double [] scale(double [] v1, double v1p);
//	
//	public static double [] plus2V(double [] v1, double v1p, double [] v2);
//	
//	public static double [] plus2V(double [] v1, double [] v2, double [] v3);
//	
//	public static double [] plus2V(double [] v1, double v1p, double [] v2, boolean reset);
//	
//	public static double [] plus2V(double [] v1, double [] v2);
//	
//	public static double [] divideElements(double [] v1, double [] v2, double [] rs);
//	
//	public static double [] constrains(double [] v1, double lowb, double upb) {  
//		for (int i = 0; i < v1.length; i++) { 
//			if (v1[i] < lowb) {
//				v1[i] = lowb;
//			}
//			if (v1[i] > upb) {
//				v1[i] = upb;
//			}
//		}
//		return v1;
//	}
//	
//	public static double [] multipleElements(double [] v1, double [] v2, double [] rs) {  
//		for (int i = 0; i < rs.length; i++) {
//			rs[i] = v1[i] * v2[i]; 
//		}
//		return rs;
//	}
//	
//	public static double [] difMultipleElements(double [] dv1, double [] v2, double [] drs) { 
//		for (int i = 0; i < dv1.length; i++) {
//			dv1[i] =  drs[i] * v2[i]; 
//		}
//		return dv1;
//	}
//	
//	public static double multiple(double [] v1, double [] v2) { 
//		double v = 0;
//		for (int i = 0; i < v1.length; i++) {
//			v = v + v1[i] * v2[i]; 
//		}
//		return v;
//	}
//	
//	public static double [] difMultiple(double dr, double [] dv, double [] v2) { 
//		for (int i = 0; i < dv.length; i++) {
//			dv[i] = dr * v2[i]; 
//		}
//		return dv;
//	}
//	
//	public static double cos(double [] v1, double [] v2) {
//		double s = 0;
//		s = Math.abs(multiple(v1, v2));
//		double a = Math.pow(multiple(v1, v1) * multiple(v2, v2), 0.5);
//		if (a == 0) {
//			return -1.0;
//		}
//		return s/a;
//	}
//	
//	public static double [] difCos(double dr, double [] dv1, double [] v1, double [] v2) {
//		double s = 0;
//		s = Math.abs(multiple(v1, v2));
//		double v1_2 = multiple(v1, v1);
//		double v2_2 = multiple(v2, v2);
//		
//		double a = Math.pow(v1_2 * v2_2, 0.5);
//		if (a == 0 || Math.abs(a) < 1.0E-7) {  
////		if (a == 0) {
//			return dv1;
//		}
//		for (int i = 0; i < dv1.length; i++) {
//			dv1[i] = dr *(v2[i]/a - 0.5 * s * Math.pow(v1_2 * v2_2, -1.5) * 2.0 * v2_2* v1[i]);
//		}
//		return dv1;
//	}
//	
//	
//	
//	public static double[] softMax(double [] x, double beta) {
//		double [] ex = new double[x.length];
//		double s = 0;
//		int mi = getMaxPos(x);
//		for (int i = 0; i < ex.length; i++) { 
//			ex[i] = Math.exp((x[i] - x[mi])* beta);
//			s = s + ex[i];
//		}
//		for (int i = 0; i < ex.length; i++) {
//			ex[i] = ex[i]/s;
//		}
//		return ex;
//	}
//	
////	public static void main(String[] args) {
////		System.out.println(Math.exp(700));
////	}
//	
//	public static double difSoftMax4Beta(double [] dr, double [] x, double beta) {
//		double db = 0;
//		double [] ex = new double[x.length];
//		double s = 0;
//		int mi = getMaxPos(x);
//		for (int i = 0; i < ex.length; i++) {
//			ex[i] = Math.exp((x[i] - x[mi])* beta);
//			s = s + ex[i];
//		} 
//		for (int i = 0; i < dr.length; i++) {
//			db = db + dr[i] * ex[i] * x[i]/s; 
//			double t = - ex[i]/(s * s);
//			for (int j = 0; j < ex.length; j++) {
//				db = db + dr[i] * t * ex[j] * x[j];				
//			}
//		}
//		
//		return db;
//	}
//	
//	public static double[] difSoftMax4Weighting(double [] dr, double [] x, double beta) {
//		double [] ex = new double[x.length];
//		double s = 0;
//		
//		int mi = getMaxPos(x);
//		for (int i = 0; i < ex.length; i++) {
//			ex[i] = Math.exp((x[i] - x[mi])* beta);
//			s = s + ex[i];
//		}
//		for (int i = 0; i < ex.length; i++) {
//			ex[i] = ex[i]/s;
//		}
//		
//		double [] dx = new double[x.length];
//		for (int i = 0; i < dr.length; i++) {
//			for (int j = 0; j < dx.length; j++) {
//				if (i == j) {
//					dx[j] = dx[j] + beta * ex[i] * (1 - ex[i]) * dr[i];
//				} else {
//					dx[j] = dx[j] - beta * ex[i] * ex[j] * dr[i];
////					dx[j] = dx[j] - beta * Math.exp((x[i] - x[mi])* beta) * Math.exp((x[j] - x[mi])* beta) /(s * s) * dr[i];
//				}
//			}
//		}
//		
//		return dx;
//	}
//	
//	public static double sigmod(double x) {
//		return 1.0/(1.0+Math.exp(-x));
//	}
//
//	public static void main(String[] args) {
//		System.out.println(onePlus(-1000000));
//	}
//	
//	public static double difSigmod(double x) {
//		return sigmod(x) * (1.0 - sigmod(x)) ;
//	}
//	
//	public static double onePlus(double x) {
////		double max = 50;
////		if (x > max) {
////			x = max;
////		}
//		return 1.0 + Math.log(1 + Math.exp(x));
//	}
//	
//	public static double difOnePlus(double x) {
//		return 1.0/(1.0+Math.exp(-x));
//	}
	
	public double [][] transpose(double [][] x);
	
	public void minus(double [][] x, double [][] y, double [][] r);
	
	public void multipleByElements(double [][] x, double [][] y, double [][] r);
	
	public void difMultipleByElements(double [][] dx, double [][] y, double [][] r);
	
	public void plus(double [][] x, double xp, double [][] y, double yp, double [][] r);
	
	public void plus(final float[][] x, final float xp, final float[][] y, final float yp, final float[][] r);
	
	public void plus(double [][] x, double [][] y, double yp, double [][] r);
	
	public void plus(float [][] x, float [][] y, float yp, float [][] r);
	
	public void set(double [][] x, double [][] y);
	
	public double[] multipleV2v(double [][] x, double [] y);
	
	public double[][] multipleV(double [][] x, double [] y);
	
	public double[][] multiple(double [][] x, double [][] y); 
	
	public double[][] multiple(double [][] x, double [][] y, double [][] r); 
	
	public float[][] multiple(float [][] x, float [][] y, float [][] r); 
	
	public double[][] difMultipleX(double [][] dr, double [][] y);
	
	public double[][] difMultipleX(double [][] dr, double [][] y, double [][] dx);
	
	public float[][] difMultipleX(float[][] dr, float[][] y, float[][] dx);
	
	public double[][] difMultipleX(double [][] dr, double [] y);
	
	public double[][] difMultipleX(double [] dr, double [] y);
	
	public double[] difMultipleY2v(double []dr, double [][] x);
	
	public double[][] difMultipleY(double []dr, double [][] x);
	
	public double[][] difMultipleY(double [][] dr, double [][] x);
	
	public double[][] difMultipleY(double [][] dr, double [][] x, double [][] dy);
	
	public float[][] difMultipleY(float [][] dr, float [][] x, float [][] dy);
	
//	public void difSimplex(double [][] x, double [][] xSum, double [][] dx);
//	
//	public void simplex(double [][] x, double [][] xSum, int k);

}
