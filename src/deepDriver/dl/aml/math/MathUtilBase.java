package deepDriver.dl.aml.math;

import java.util.Random;

import deepDriver.dl.aml.random.RandomFactory;

public class MathUtilBase {
	
	public static double sum(double [] v1) { 
		double sum = 0;
		for (int i = 0; i < v1.length; i++) {
			 sum = sum + v1[i] ; 
		}
		return sum;
	}
	
	public static double sum(double [][] x, int ci) { 
		double sum = 0;
		for (int i = 0; i < x.length; i++) {
			 sum = sum + x[i][ci] ; 
		}
		return sum;
	}
	
	public static double [] plus(double [] v1, double v1p) { 
		for (int i = 0; i < v1.length; i++) {
			v1[i] = v1p + v1[i]; 
		}
		return v1;
	} 
	
	public static double checkNormal(double [] x) {
		double s = 0;
		for (int i = 0; i < x.length; i++) {
			if (x[i] < 0) {
				return -1;
			} 
			s = s + x[i];
		}		
		return s;
	}
	
	public static double normalize(double [] x) {
		double s = 0;
		for (int i = 0; i < x.length; i++) {
			s = s + x[i];
		}	
		if(s == 0) {
			return s;
		}
		for (int i = 0; i < x.length; i++) {
			x[i] = x[i]/s;
		}
		return s;
	}
	
	public static double [][] allocateE(int r, int c) { 
		double [][] m = new double[r][c];
		for (int i = 0; i < m.length; i++) {
			m[i] = new double[c];
			for (int j = 0; j < m[i].length; j++) {
				m[i][j] = 1.0;
			}
		}
		return m;
	}
	
	public static Random random = RandomFactory.getRandom();
	
	public static void initMatrix(double [][] x, double length, double min) {
		for (int i = 0; i < x.length; i++) {
			for (int j = 0; j < x[i].length; j++) {
				x[i][j] = length * random.nextDouble()
						+ min;
			}
		}
	}
	
	public static int [][] allocateInt(int r, int c) { 
		int [][] m = new int[r][c];
		for (int i = 0; i < m.length; i++) {
			m[i] = new int[c];
		}
		return m;
	}
	
	public static void reset(boolean [] bs) { 
		for (int i = 0; i < bs.length; i++) {
			bs[i] = false;
		}
	}
	
	
	public static double [][] allocate(int r, int c) { 
		double [][] m = new double[r][c];
		for (int i = 0; i < m.length; i++) {
			m[i] = new double[c];
		}
		return m;
	}
	
	public static void reset2zero(double [] x) {  
		for (int i = 0; i < x.length; i++) {
			x[i] = 0;
		}
	}
	
	public static void reset2zero(double [][][] x) {  
		for (int i = 0; i < x.length; i++) {
			reset2zero(x[i]);
		}
	}
	
	public static void reset2zero(double [][] x) {  
		for (int i = 0; i < x.length; i++) {
			for (int j = 0; j < x[i].length; j++) {
				x[i][j] = 0;
			}
		}
	}
	
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
	
	public static double [] plus2V(double [] v1, double [] v2, double [] v3) { 
		for (int i = 0; i < v3.length; i++) {
			v3[i] = v1[i] +  v2[i]; 
		}
		return v3;
	}
	
	public static double [] plus2V(double [] v1, double v1p, double [] v2, boolean reset) { 
		for (int i = 0; i < v2.length; i++) {
			double b = v2[i]; 
			if (reset) {
				b = 0;
			} 
			v2[i] = v1p * v1[i] +  b;
		}
		return v2;
	}
	
	public static double [] plus2V(double [] v1, double [] v2) { 
		return plus2V(v1, 1.0, v2); 
	}
	
	public static double [] divideElements(double [] v1, double [] v2, double [] rs) {  
		for (int i = 0; i < rs.length; i++) {
			if (v2[i] != 0) {
				rs[i] = v1[i] / v2[i]; 
			}
			
		}
		return rs;
	}
	
	public static double [] constrains(double [] v1, double lowb, double upb) {  
		for (int i = 0; i < v1.length; i++) { 
			if (v1[i] < lowb) {
				v1[i] = lowb;
			}
			if (v1[i] > upb) {
				v1[i] = upb;
			}
		}
		return v1;
	}
	
	public static double [] multipleElements(double [] v1, double [] v2, double [] rs) {  
		for (int i = 0; i < rs.length; i++) {
			rs[i] = v1[i] * v2[i]; 
		}
		return rs;
	}
	
	public static double [] difMultipleElements(double [] dv1, double [] v2, double [] drs) { 
		for (int i = 0; i < dv1.length; i++) {
			dv1[i] =  drs[i] * v2[i]; 
		}
		return dv1;
	}
	
//	public static float multipleNative(float [] v1, float [] v2) {   
//		FloatMatrix fm = new FloatMatrix(v1);
//		FloatMatrix fm1 = new FloatMatrix(v2);
////		float v = 0;
////		for (int i = 0; i < v1.length; i++) {
////			v = v + v1[i] * v2[i]; 
////		}
//		return fm.mul(fm1.transpose()).data[0];
//	}
	
	public static float multiple(float [] v1, float [] v2) { 
		float v = 0;
		for (int i = 0; i < v1.length; i++) {
			v = v + v1[i] * v2[i]; 
		}
		return v;
	}
	
	public static double multiple(double [] v1, double [] v2) { 
		double v = 0;
		for (int i = 0; i < v1.length; i++) {
			v = v + v1[i] * v2[i]; 
		}
		return v;
	}
	
	public static double [] difMultiple(double dr, double [] dv, double [] v2) { 
		for (int i = 0; i < dv.length; i++) {
			dv[i] = dr * v2[i]; 
		}
		return dv;
	}
	
	public static double cos(double [] v1, double [] v2) {
		double s = 0;
		s = multiple(v1, v2);
		double a = Math.pow(multiple(v1, v1) * multiple(v2, v2), 0.5);
		if (a == 0) {
			return -1.0;
		}
		return s/a;
	}
	
	public static double [] difCos(double dr, double [] dv1, double [] v1, double [] v2) {
		double s = 0;
		s = multiple(v1, v2);
		double v1_2 = multiple(v1, v1);
		double v2_2 = multiple(v2, v2);
		
		double a = Math.pow(v1_2 * v2_2, 0.5);
		if (a == 0 || Math.abs(a) < 1.0E-7) {  
//		if (a == 0) {
			return dv1;
		}
		for (int i = 0; i < dv1.length; i++) {
			dv1[i] = dr *(v2[i]/a - 0.5 * s * Math.pow(v1_2 * v2_2, -1.5) * 2.0 * v2_2* v1[i]);
		}
		return dv1;
	}
	
	
	
	public static double[] softMax(double [] x, double beta) {
		double [] ex = new double[x.length];
		double s = 0;
		int mi = getMaxPos(x);
		for (int i = 0; i < ex.length; i++) { 
			ex[i] = Math.exp((x[i] - x[mi])* beta);
			s = s + ex[i];
		}
		for (int i = 0; i < ex.length; i++) {
			ex[i] = ex[i]/s;
		}
		return ex;
	}
	
//	public static void main(String[] args) {
//		System.out.println(Math.exp(700));
//	}
	
	public static double difSoftMax4Beta(double [] dr, double [] x, double beta) {
		double db = 0;
		double [] ex = new double[x.length];
		double s = 0;
		int mi = getMaxPos(x);
		for (int i = 0; i < ex.length; i++) {
			ex[i] = Math.exp((x[i] - x[mi])* beta);
			s = s + ex[i];
		} 
		for (int i = 0; i < dr.length; i++) {
			db = db + dr[i] * ex[i] * x[i]/s; 
			double t = - ex[i]/(s * s);
			for (int j = 0; j < ex.length; j++) {
				db = db + dr[i] * t * ex[j] * x[j];				
			}
		}
		
		return db;
	}
	
	public static double[] difSoftMax4Weighting(double [] dr, double [] x, double beta) {
		double [] ex = new double[x.length];
		double s = 0;
		
		int mi = getMaxPos(x);
		for (int i = 0; i < ex.length; i++) {
			ex[i] = Math.exp((x[i] - x[mi])* beta);
			s = s + ex[i];
		}
		for (int i = 0; i < ex.length; i++) {
			ex[i] = ex[i]/s;
		}
		
		double [] dx = new double[x.length];
		for (int i = 0; i < dr.length; i++) {
			for (int j = 0; j < dx.length; j++) {
				if (i == j) {
					dx[j] = dx[j] + beta * ex[i] * (1 - ex[i]) * dr[i];
				} else {
					dx[j] = dx[j] - beta * ex[i] * ex[j] * dr[i];
//					dx[j] = dx[j] - beta * Math.exp((x[i] - x[mi])* beta) * Math.exp((x[j] - x[mi])* beta) /(s * s) * dr[i];
				}
			}
		}
		
		return dx;
	}
	
	public static double sigmod(double x) {
		return 1.0/(1.0+Math.exp(-x));
	}

	public static void main(String[] args) {
		System.out.println(onePlus(-1000000));
	}
	
	public static double difSigmod(double x) {
		return sigmod(x) * (1.0 - sigmod(x)) ;
	}
	
	public static double onePlus(double x) {
//		double max = 50;
//		if (x > max) {
//			x = max;
//		}
		return 1.0 + Math.log(1 + Math.exp(x));
	}
	
	public static double difOnePlus(double x) {
		return 1.0/(1.0+Math.exp(-x));
	}
	
	public static float [][] transpose(float [][] x) {
		float [][] t = new float[x[0].length][];
		for (int i = 0; i < t.length; i++) {
			t[i] = new float[x.length];
			for (int j = 0; j < t[i].length; j++) {
				t[i][j] = x[j][i];
			}
		}
		return t;
	}
	
	public static double [][] transpose(double [][] x) {
		double [][] t = new double[x[0].length][];
		for (int i = 0; i < t.length; i++) {
			t[i] = new double[x.length];
			for (int j = 0; j < t[i].length; j++) {
				t[i][j] = x[j][i];
			}
		}
		return t;
	}
	
	public static void minus(double [][] x, double [][] y, double [][] r) { 
		for (int i = 0; i < x.length; i++) { 
			for (int j = 0; j < x[i].length; j++) {
				r[i][j] = x[i][j] - y[i][j];
			}
		} 
	}
	
	public static void multipleByElements(double [][] x, double [][] y, double [][] r) { 
		for (int i = 0; i < x.length; i++) { 
			for (int j = 0; j < x[i].length; j++) {
				r[i][j] = x[i][j] * y[i][j];
			}
		} 
	}
	
	public static void difMultipleByElements(double [][] dx, double [][] y, double [][] r) { 
		for (int i = 0; i < dx.length; i++) { 
			for (int j = 0; j < dx[i].length; j++) {
				r[i][j] = dx[i][j] * y[i][j];
			}
		} 
	}
	
	public static void plus(double [][] x, double [][] y, double [][] r) {
		plus(x, y, 1.0, r);
	}
	
	public static void plus(double [][] x, double xp, double [][] y, double yp, double [][] r) { 
		for (int i = 0; i < x.length; i++) { 
			for (int j = 0; j < x[i].length; j++) {
				r[i][j] = xp * x[i][j] + yp * y[i][j];
			}
		} 
	}	
	
	public static void plus(float [][] x, float xp, float [][] y, float yp, float [][] r) { 
		for (int i = 0; i < x.length; i++) { 
			for (int j = 0; j < x[i].length; j++) {
				r[i][j] = xp * x[i][j] + yp * y[i][j];
			}
		} 
	}
	
	
	public static void plus(float [][] x, float [][] y, float yp, float [][] r) { 
		for (int i = 0; i < x.length; i++) { 
			for (int j = 0; j < x[i].length; j++) {
				r[i][j] = x[i][j] + yp * y[i][j];
			}
		} 
	}
	
	public static void plus(double [][] x, double [][] y, double yp, double [][] r) { 
		for (int i = 0; i < x.length; i++) { 
			for (int j = 0; j < x[i].length; j++) {
				r[i][j] = x[i][j] + yp * y[i][j];
			}
		} 
	}
	
	public static void set(double [][] x, double [][] y) { 
		for (int i = 0; i < x.length; i++) { 
			for (int j = 0; j < x[i].length; j++) {
				x[i][j] = y[i][j];
			}
		} 
	}
	
	public static double [] matrix2Vector(double [][] m) {
		double [] v = new double[m.length];
		for (int i = 0; i < v.length; i++) {
			v[i] = m[i][0];
		}
		return v;
	}
	
	public static double[] multipleV2v(double [][] x, double [] y) {
		double[][] r = multipleV(x, y);
		return matrix2Vector(r);
	}
	/**
	 * Take the double [] y as the vertical vector, not a horizetal one.
	 * no need tranpose again and again. 
	 * **/
	public static double[][] multipleV(double [][] x, double [] y) {
		double[][] result = new double[x.length][];
		for (int i = 0; i < result.length; i++) {
			result[i] = new double[1];
			for (int j = 0; j < result[i].length; j++) {
				result[i][j] = multiple(x[i], y);
			}
		}
		return result;
	}
	
	public static double[][] multiple(double [][] x, double [][] y) {
		double[][] result = new double[x.length][];
		return multiple(x, y, result);
	}
	
	public static float[][] multiple(float [][] x, float [][] y, float [][] result) {
		float [][] t = transpose(y);		
		for (int i = 0; i < result.length; i++) {
			if (result[i] == null) {
				result[i] = new float[t.length];
			}			
			for (int j = 0; j < result[i].length; j++) {
				result[i][j] = multiple(x[i], t[j]);
			}
		}
		return result;
	}
	
	public static double[][] multiple(double [][] x, double [][] y, double [][] result) {
		double [][] t = transpose(y);		
		for (int i = 0; i < result.length; i++) {
			if (result[i] == null) {
				result[i] = new double[t.length];
			}			
			for (int j = 0; j < result[i].length; j++) {
				result[i][j] = multiple(x[i], t[j]);
			}
		}
		return result;
	}
	
	/***
	 * Z = X * Y
	 * 
	 * ***/
	public static double[][] difMultipleX(double [][] dr, double [][] y) {
		double [][] dm = new double[dr.length][];
		return difMultipleX(dr, y, dm);
	}
	
	public static float[][] difMultipleX(float [][] dr, float [][] y, float [][] dm) {
		//dr ith row, y jth row		
		for (int i = 0; i < dm.length; i++) {
			if (dm[i] == null) {
				dm[i] = new float[y.length];
			}			
			for (int j = 0; j < dm[i].length; j++) {
				dm[i][j] = multiple(dr[i] , y[j]);
			}
		}
		return dm;
	}
	
	public static double[][] difMultipleX(double [][] dr, double [][] y, double [][] dm) {
		//dr ith row, y jth row		
		for (int i = 0; i < dm.length; i++) {
			if (dm[i] == null) {
				dm[i] = new double[y.length];
			}			
			for (int j = 0; j < dm[i].length; j++) {
				dm[i][j] = multiple(dr[i] , y[j]);
			}
		}
		return dm;
	}
	
	public static double[][] difMultipleX(double [][] dr, double [] y) {
		//dr ith row, y jth row
		double [][] yt = transpose(new double[][]{y});
		return difMultipleX(dr, yt);
	}
	
	public static double[][] difMultipleX(double [] dr, double [] y) {
		//dr ith row, y jth row
		double [][] drt = transpose(new double[][]{dr});
		double [][] yt = transpose(new double[][]{y});
		return difMultipleX(drt, yt);
	}
	
	public static double[] difMultipleY2v(double []dr, double [][] x) {
		double [][] y = difMultipleY(dr, x);
		return matrix2Vector(y);
	}
	
	public static double[][] difMultipleY(double []dr, double [][] x) {
		double [][] drt = transpose(new double[][]{dr});
		return difMultipleY(drt, x);
	}
	
	public static double[][] difMultipleY(double [][] dr, double [][] x) {
		double [][] dm = new double[x[0].length][];
		return difMultipleY(dr, x, dm);
	}
	
	public static float[][] difMultipleY(float [][] dr, float [][] x, float [][] dm) {
		//dr jth column, x ith column
		float [][] drt = transpose(dr);
		float [][] xt = transpose(x);
		
		for (int i = 0; i < dm.length; i++) {
			if (dm[i] == null) {
				dm[i] = new float[dr[0].length];
			}
			
			for (int j = 0; j < dm[i].length; j++) {
				dm[i][j] = multiple(drt[j] , xt[i]);
			}
		}
		return dm;
	}
	
	public static double[][] difMultipleY(double [][] dr, double [][] x, double [][] dm) {
		//dr jth column, x ith column
		double [][] drt = transpose(dr);
		double [][] xt = transpose(x);
		
		for (int i = 0; i < dm.length; i++) {
			if (dm[i] == null) {
				dm[i] = new double[dr[0].length];
			}
			
			for (int j = 0; j < dm[i].length; j++) {
				dm[i][j] = multiple(drt[j] , xt[i]);
			}
		}
		return dm;
	}
		
	public static boolean check(double [] ta, double [] tb) {
		if (getMaxPos(ta) == getMaxPos(tb)) {
			return true;
		}
		return false;
	}
	
	public static boolean check(double [] ta, int mxPos) {
		if (getMaxPos(ta) == mxPos) {
			return true;
		}
		return false;
	}
	
	public static int getMaxPos(double [] ta) {
		int pos = 0;
		for (int i = 0; i < ta.length; i++) {
			if (ta[i] > ta[pos]) {
				pos = i;
			}
		}
		return pos;
	}
	
	public static boolean isNaN(double [] x) {
		for (int i = 0; i < x.length; i++) {
			if (isNaN(x[i])) {
				return true;
			}
		}
		return false;
	}
	
	public static boolean isNaN(double a) {
		if (!(a > 0 || a <= 0)) {
			return true;
		} 
		return false;
	}	
	
	static boolean useSimplex = true;
	
	public static void gm(double [] x, double gm) {
		double s = 0;
		for (int i = 0; i < x.length; i++) {
			s = s + x[i] * x[i];
		}
		if (s == 0) {
			return ;
		}
		s = Math.sqrt(s);
		if (s > gm) {
			for (int i = 0; i < x.length; i++) {
				x[i] =  x[i] * gm / s;
			}
		}				
	}
	
	public static double [] difSimplex(double [] x, double s, double [] dx) {
		if (!useSimplex) {
			return dx;
		}
		if (s == 0) {
			return dx;
		}
		for (int i = 0; i < dx.length; i++) {
			if (x[i] > 0) {
				dx[i] = dx[i]/s;
			}
			if (x[i] == 0) {
				dx[i] = 0;
			}
		}
		return dx;
	}
	
	/*public static double [] simplex(double [] x, int k) {
		return simplex(x, -1, k);
	}**/
	
	public static double [] simplexDep(double [] x, double s, int k) {
		if (!useSimplex) {
			return x;
		}
		if (s < 0) {
			s = sum(x);
		}
		if (s == 0) {
			return x;
		}
		for (int i = 0; i < x.length; i++) {
			x[i] = x[i]/s;
		}
		if (k > x.length) {
			return x;
		}
		int [] ids = sort(x);
		for (int i = 0; i < ids.length - k; i++) {
			x[ids[i]] = 0;
		}
		return x;
	}
	
	public static double [] simplex2(double [] x, double s, int k) {
		if (!useSimplex) {
			return x;
		}
		if (s < 0) {
			s = sum(x);
		}
		if (s == 0) {
			return x;
		}
		for (int i = 0; i < x.length; i++) {
			x[i] = x[i]/s;
		} 
		int [] ids = sort(x);
		for (int i = 0; i < ids.length - k; i++) {
			x[ids[i]] = 0;
		}
		double _1_k = 1.0/(double)k;
		for (int i = 0; i < x.length; i++) {
			if (x[i] < _1_k) {
				x[i] = 0;
			}
		}
		return x;
	}
	
	public static double sumMaxK(double [] x,  int k) {
		int [] ids = sort(x);
		if (k > ids.length ) {
			k = ids.length;
		}
		double s = 0;
		for (int i = ids.length - 1; (ids.length - 1 - i + 1) <= k; i--) {
			s = s + x[ids[i]];
		}
		return s;
	}
	
	public static int [] sort(double [] x) {		
		int [] id = new int[x.length];
		boolean [] fid = new boolean[id.length];
		int cnt = 0; 
		for (int i = 0; i < id.length; i++) {
			int mi = 0;
			double min = x[mi];
			for (int j = 0; j < x.length; j++) {
				if (fid[j]) {
					continue;
				}
				if (fid[mi]) {
					mi = j;
					min = x[j]; 
				} else { 
					if (min > x[j]) {
						mi = j;
						min = x[j]; 
					}					
				}
			}
			fid[mi] = true;
			id[cnt ++] = mi;
		}		
		return id;
	}
	
		
	public static int K = 8;
	public static double _1_K = 1.0/(double)K;
	
	public static void difSimplex(double [][] x, double [][] xSum, double [][] dx) {
		if (!useSimplex) {
			return;
		}
		/***Disable column simplex.
		double [][] tx = transpose(x);
		double [][] tdx = transpose(dx);
		for (int i = 0; i < tdx.length; i++) {
			difSimplex(tx[i], xSum[0][0], tdx[i]);
		}
		double [][] tdx2 = transpose(tdx);
		set(dx, tdx2);*****/
		
		for (int i = 0; i < dx.length; i++) {
			difSimplex(x[i], xSum[0][0], dx[i]);
		}
	}
	
	public static void simplex(double [][] x, double [][] xSum, int k) {
//		xSum = new double[2][];
		if (!useSimplex) {
			return;
		}
		double _1_k = 1.0/(double)k;
		if (xSum[0] == null) {
			xSum[0] = new double[x.length];
		}
		if (xSum[1] == null) {
			xSum[1] = new double[x[0].length];
		}
		for (int i = 0; i < x.length; i++) {
			xSum[0][i] = sumMaxK(x[i], k);
			if (xSum[0][0] < xSum[0][i]) {
				xSum[0][0] = xSum[0][i];
			}
		}
		
		for (int i = 0; i < x.length; i++) {
//			xSum[0][i] = sumMaxK(x[i], k);
			simplex2(x[i], xSum[0][0], k);
		}
		
		/***Disable column simplex.
		double [][] tx = transpose(x);
		for (int i = 0; i < tx.length; i++) {
//			xSum[1][i] = sum(tx[i]);
			simplex2(tx[i], xSum[0][0], _1_k);
//			if (xSum[1][i] > 1.0) {
//				simplex(tx[i], k);
//			} else {
//				xSum[1][i] = 1.0;
//			}
		}
		double [][] tx2 = transpose(tx);
		set(x, tx2);***/
	}
	
	
	

}
