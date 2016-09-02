package deepDriver.dl.aml.attention;

import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.random.RandomFactory;

public class SoftAttention {
	
	IActivationFunction af;
	
	double [] wa;
	double [] dwa;	
	double [] ua;
	double [] dua;
	double va;
	double dva;
	
	int maxLength;
	double [][] as;
	double [][] afs;
	
	double [][] es;
//	double [][] des;
	double [][] das;
	
	public SoftAttention(int waSize, int uaSize, IActivationFunction af, int maxLength) {
		wa = new double[waSize];
		dwa = new double[wa.length];
		ua = new double[uaSize];
		dua = new double[ua.length];
		for (int i = 0; i < wa.length; i++) {
			wa[i] = RandomFactory.getRandom().nextDouble();			
		}
		for (int i = 0; i < ua.length; i++) {
			ua[i] = RandomFactory.getRandom().nextDouble();			
		}
		va = RandomFactory.getRandom().nextDouble();
		this.af = af;
		this.maxLength = maxLength;
		
		as = new double[maxLength][maxLength];
		afs = new double[maxLength][maxLength];
		es = new double[maxLength][maxLength];
		das = new double[maxLength][maxLength];
		for (int i = 0; i < as.length; i++) {
			as[i] = new double[maxLength];
			afs[i] = new double[maxLength];
			es[i] = new double[maxLength];
			das[i] = new double[maxLength];
		}
	}
	
	double l;
	double m; 
	
	public double getL() {
		return l;
	}

	public void setL(double l) {
		this.l = l;
	}

	public double getM() {
		return m;
	}

	public void setM(double m) {
		this.m = m;
	}
	
	public void updateWw() {
		for (int i = 0; i < wa.length; i++) {
			wa[i] = wa[i] - l * dwa[i];
		}
		for (int i = 0; i < ua.length; i++) {
			ua[i] = ua[i] - l * dua[i];
		}
		va = va - l * dva;
	}

	public double dot(double [] a, double [] b) {
		double dot = 0;
		for (int i = 0; i < b.length; i++) {			
			dot = dot + a[i] * b[i];
		}
		return dot;
	}
	
	double [][] hj;
	double [][] dhj;
	double [] dsc;
	
	public double[][] getDhj() {
		return dhj;
	}

	public double[] getDsc() {
		return dsc;
	}	 
	
//	double [] wc;
//	double [] dwc;
//	
//	public double forwardSt(double [] st_1, double [][] hj, int t) {
//		double [] c = forward(st_1, hj, t);
//		if (wc == null) {
//			wc = new double[c.length];
//			dwc = new double[wc.length];
//			for (int i = 0; i < wc.length; i++) {
//				wc[i] = RandomFactory.getRandom().nextDouble(); 
//			}
//		}
//		return dot(wc, c);
//	}

	public double [] forward(double [] st_1, double [][] hj, int t) {
		if (t == 0) {			
			this.hj = hj;
			dhj = new double[hj.length][];
			for (int i = 0; i < dhj.length; i++) {
				dhj[i] = new double[hj[i].length];
			}			
			double [] das1 = das[t];
			for (int i = 0; i < das1.length; i++) {
				das1[i] = 0;
			}
			dva = 0;
			for (int i = 0; i < dua.length; i++) {
				dua[i] = 0;
			}
			for (int i = 0; i < dwa.length; i++) {
				dwa[i] = 0;
			}
		}
		int hLength = hj.length;
		double [] ast = as[t];
		double [] est = es[t];
		double [] afst = afs[t];
		double scDot = dot(wa, st_1);
		for (int i = 0; i < hLength; i++) {
			afst[i] = scDot + dot(ua, hj[i]);
			ast[i] = va * af.activate(afst[i]);
		}
		double esSum = 0;
		
		double maxAs = 0;
		for (int i = 0; i < hLength; i++) {
			if (maxAs < ast[i]) {
				maxAs = ast[i];
			}
		}
		for (int i = 0; i < hLength; i++) {
			est[i] = Math.exp(ast[i] - maxAs);
			esSum = esSum + est[i];
		}
		
		for (int i = 0; i < hLength; i++) {
			est[i] = est[i]/esSum; 
		}
		
		double [] c = new double[ua.length];
		for (int i = 0; i < hj.length; i++) {
			for (int j = 0; j < hj[i].length; j++) {
				c[j] = c[j] + est[i] * hj[i][j];
			}
		}
		return c;
	}
	

	public void bp(double [] st_1, double [] dc, int t) {
//		double [] ast = as[t];
		double [] dast = das[t];
		double [] est = es[t];
		double [] dest = new double[es.length];
		double [] afst = afs[t];
		int djLength = dhj.length;
		
		dsc = new double[st_1.length];
		
		for (int i = 0; i < dhj.length; i++) {
			for (int j = 0; j < dhj[i].length; j++) {
				dhj[i][j] = dhj[i][j] + dc[j] * est[i];
				dest[i] = dest[i] + dc[j] * hj[i][j];
			}
		}
		
		for (int i = 0; i < djLength; i++) {//dest
			for (int j = 0; j < djLength; j++) {//dast
				if (i == j) {
					dast[j] = dast[j] + dest[i] * ((est[i]) - (est[i]) * (est[i]));
				} else {
					dast[j] = dast[j] - dest[i] * (- (est[i]) * (est[j]));
				}
			}
		}
		
		for (int i = 0; i < djLength; i++) {
			dva = dva + dast[i] * af.activate(afst[i]);
			double dz = dast[i] * va * af.deActivate(afst[i]);
			for (int j = 0; j < dwa.length; j++) {
				dwa[j] = dwa[j] + dz * st_1[j];
				dsc[j] = dsc[j] + dz * wa[j];
			}
			for (int j = 0; j < dua.length; j++) {
				dua[j] = dua[j] + dz * hj[i][j];
				dhj[i][j] = dhj[i][j] + dz * ua[j];
			}
		}
		
	}

}
