package deepDriver.dl.aml.math;

import deepDriver.dl.aml.distribution.modelParallel.PartialCallback;
import deepDriver.dl.aml.distribution.modelParallel.ThreadParallel;

public class MathUtil4MThreads implements IMathFunction {
	
	ThreadParallel tp = new ThreadParallel();
	int threadCnt = 1;
	
	public int getThreadCnt() {
		return threadCnt;
	}

	public void setThreadCnt(int threadCnt) {
		this.threadCnt = threadCnt;
	}
	
	public float[][] transposePartial(float [][] t, float[][] x, int offset, int runLen) { 		
		for (int i = offset; i < offset + runLen; i++) {
			t[i] = new float[x.length];
			for (int j = 0; j < t[i].length; j++) {
				t[i][j] = x[j][i];
			}
		}
		return t;
	}

	public double[][] transposePartial(double [][] t, double[][] x, int offset, int runLen) { 		
		for (int i = offset; i < offset + runLen; i++) {
			t[i] = new double[x.length];
			for (int j = 0; j < t[i].length; j++) {
				t[i][j] = x[j][i];
			}
		}
		return t;
	}
	
	public float[][] transpose(final float[][] x) {
		if (threadCnt == 1) {
			return MathUtilBase.transpose(x);
		}
		
		final float [][] t = new float[x[0].length][];
		tp.runMutipleThreads(t.length, new PartialCallback() {
			
			@Override
			public void runPartial(int offset, int runLen) {
				transposePartial(t, x, offset, runLen);	
			}
		}, threadCnt);
		return t;
	}

	@Override
	public double[][] transpose(final double[][] x) {
		if (threadCnt == 1) {
			return MathUtilBase.transpose(x);
		}
		
		final double [][] t = new double[x[0].length][];
		tp.runMutipleThreads(t.length, new PartialCallback() {
			
			@Override
			public void runPartial(int offset, int runLen) {
				transposePartial(t, x, offset, runLen);	
			}
		}, threadCnt);
		return t;
	}
	
	public void minusPartial(double[][] x, double[][] y, double[][] r, int offset, int runLen) {
		for (int i = offset; i < offset + runLen; i++) { 
			for (int j = 0; j < x[i].length; j++) {
				r[i][j] = x[i][j] - y[i][j];
			}
		} 
	}

	@Override
	public void minus(final double[][] x, final double[][] y, final double[][] r) {
		if (threadCnt == 1) {
			MathUtilBase.minus(x, y, r);
			return;
		}
		
		tp.runMutipleThreads(r.length, new PartialCallback() {				
				@Override
				public void runPartial(int offset, int runLen) {
					minusPartial(x, y, r, offset, runLen);	
				}
			}, threadCnt);		
	}
	
	public void multipleByElementsPartial(double[][] x, double[][] y, double[][] r, int offset, int runLen) { 
		for (int i = offset; i < offset + runLen; i++) { 
			for (int j = 0; j < x[i].length; j++) {
				r[i][j] = x[i][j] * y[i][j];
			}
		} 
	}

	@Override
	public void multipleByElements(final double[][] x, final double[][] y, final double[][] r) {
		if (threadCnt == 1) {
			MathUtilBase.multipleByElements(x, y, r);
			return;
		}
		
		tp.runMutipleThreads(r.length, new PartialCallback() {			 
			public void runPartial(int offset, int runLen) {
				multipleByElementsPartial(x, y, r, offset, runLen);	
			}
		}, threadCnt);		
	}
	
	public void difMultipleByElementsPartial(double[][] dx, double[][] y, double[][] r, int offset, int runLen) {
		for (int i = offset; i < offset + runLen; i++) { 
			for (int j = 0; j < dx[i].length; j++) {
				r[i][j] = dx[i][j] * y[i][j];
			}
		}
	}

	@Override
	public void difMultipleByElements(final double[][] dx, final double[][] y, final double[][] r) {
		if (threadCnt == 1) {
			MathUtilBase.difMultipleByElements(dx, y, r);
			return;
		}
		
		tp.runMutipleThreads(r.length, new PartialCallback() {			 
			public void runPartial(int offset, int runLen) {
				difMultipleByElementsPartial(dx, y, r, offset, runLen);	
			}
		}, threadCnt);
	}
	
	public void plusPartial(float[][] x, float xp, float[][] y, float yp, float[][] r, int offset, int runLen) {
		for (int i = offset; i < offset + runLen; i++) { 
			for (int j = 0; j < x[i].length; j++) {
				r[i][j] = xp * x[i][j] + yp * y[i][j];
			}
		} 
	}
	
	public void plusPartial(double[][] x, double xp, double[][] y, double yp, double[][] r, int offset, int runLen) {
		for (int i = offset; i < offset + runLen; i++) { 
			for (int j = 0; j < x[i].length; j++) {
				r[i][j] = xp * x[i][j] + yp * y[i][j];
			}
		} 
	}
	
	public void plus(final float[][] x, final float xp, final float[][] y, final float yp, final float[][] r) {
		if (threadCnt == 1) {
			MathUtilBase.plus(x, xp, y, yp, r);
			return;
		}
		
		tp.runMutipleThreads(r.length, new PartialCallback() {			 
			public void runPartial(int offset, int runLen) {
				plusPartial(x, xp, y, yp, r, offset, runLen);	
			}
		}, threadCnt);
	}
	
//	@Override
//	public void plus(final float[][] x, final float xp, final float[][] y, final float yp, final float[][] r) {
//		if (threadCnt == 1) {
//			MathUtilBase.plus(x, xp, y, yp, r);
//			return;
//		}
//		
//		tp.runMutipleThreads(r.length, new PartialCallback() {			 
//			public void runPartial(int offset, int runLen) {
//				plusPartial(x, xp, y, yp, r, offset, runLen);	
//			}
//		}, threadCnt);
//	}

	@Override
	public void plus(final double[][] x, final double xp, final double[][] y, final double yp, final double[][] r) {
		if (threadCnt == 1) {
			MathUtilBase.plus(x, xp, y, yp, r);
			return;
		}
		
		tp.runMutipleThreads(r.length, new PartialCallback() {			 
			public void runPartial(int offset, int runLen) {
				plusPartial(x, xp, y, yp, r, offset, runLen);	
			}
		}, threadCnt);
	}
	
	public void plus(float[][] x, float[][] y, float yp, float[][] r) {
		if (threadCnt == 1) {
			MathUtilBase.plus(x, y, yp, r);
			return;
		}
		plus(x, 1.0f, y, yp, r);		
	}

	@Override
	public void plus(double[][] x, double[][] y, double yp, double[][] r) {
		if (threadCnt == 1) {
			MathUtilBase.plus(x, y, yp, r);
			return;
		}
		plus(x, 1.0, y, yp, r);		
	}
	
	public void setPartial(double[][] x, double[][] y, int offset, int runLen) {
		for (int i = offset; i < offset + runLen; i++) { 
			for (int j = 0; j < x[i].length; j++) {
				x[i][j] = y[i][j];
			}
		} 
	}

	@Override
	public void set(final double[][] x, final double[][] y) {
		if (threadCnt == 1) {
			MathUtilBase.set(x, y);
			return;
		}
		
		tp.runMutipleThreads(x.length, new PartialCallback() {			 
			public void runPartial(int offset, int runLen) {
				setPartial(x, y, offset, runLen);	
			}
		}, threadCnt);		
	}
	
//	public double[] multipleV2vPartial(double[][] x, double[] y) {
//		
//	}

	@Override
	public double[] multipleV2v(double[][] x, double[] y) {
		if (threadCnt == 1) {			
			return MathUtilBase.multipleV2v(x, y);
		}
		double[][] r = multipleV(x, y);
		return MathUtilBase.matrix2Vector(r);
	}
	
	public double[][] multipleVPartial(double[][] result, double[][] x, double[] y, int offset, int runLen) {		
		for (int i = offset; i < offset + runLen; i++) {
			result[i] = new double[1];
			for (int j = 0; j < result[i].length; j++) {
				result[i][j] = MathUtilBase.multiple(x[i], y);
			}
		}
		return result;
	}

	@Override
	public double[][] multipleV(final double[][] x, final double[] y) {
		if (threadCnt == 1) {			
			return MathUtilBase.multipleV(x, y);
		}
		
		final double[][] result = new double[x.length][];
		tp.runMutipleThreads(result.length, new PartialCallback() {			 
			public void runPartial(int offset, int runLen) {
				multipleVPartial(result, x, y, offset, runLen);	
			}
		}, threadCnt);		
		return result;
	}
	
	public float[][] multiplePartial(float[][] result, float[][] x, float[][] y, int offset, int runLen) {
		float [][] t = transpose(y);		
		for (int i = offset; i < offset + runLen; i++) {
			if (result[i] == null) {
				result[i] = new float[t.length];
			}
			for (int j = 0; j < result[i].length; j++) {
				result[i][j] = MathUtilBase.multiple(x[i], t[j]);
			}
		}
		return result;
	}
	
	public double[][] multiplePartial(double[][] result, double[][] x, double[][] y, int offset, int runLen) {
		double [][] t = transpose(y);		
		for (int i = offset; i < offset + runLen; i++) {
			if (result[i] == null) {
				result[i] = new double[t.length];
			}
			for (int j = 0; j < result[i].length; j++) {
				result[i][j] = MathUtilBase.multiple(x[i], t[j]);
			}
		}
		return result;
	}
	
	public double[][] multiple(double [][] x, double [][] y) {
		final double[][] result = new double[x.length][];
		return multiple(x, y, result);
	}
	
	public float[][] multiple(final float [][] x, final float [][] y, final float [][] r) {
		if (threadCnt == 1) {			
			return MathUtilBase.multiple(x, y, r);
		}	
		
		tp.runMutipleThreads(r.length, new PartialCallback() {			 
			public void runPartial(int offset, int runLen) {
				multiplePartial(r, x, y, offset, runLen);	
			}
		}, threadCnt);
		return r;
	}

	@Override
	public double[][] multiple(final double[][] x, final double[][] y, final double [][] r) {
		if (threadCnt == 1) {			
			return MathUtilBase.multiple(x, y, r);
		}	
		
		tp.runMutipleThreads(r.length, new PartialCallback() {			 
			public void runPartial(int offset, int runLen) {
				multiplePartial(r, x, y, offset, runLen);	
			}
		}, threadCnt);
		return r;
	}
	
	public float[][] difMultipleXPartial(float [][] dm, float[][] dr, float[][] y, int offset, int runLen) {		
		for (int i = offset; i < offset + runLen; i++) {
			if (dm[i] == null) {
				dm[i] = new float[y.length];
			}			
			for (int j = 0; j < dm[i].length; j++) {
				dm[i][j] = MathUtilBase.multiple(dr[i] , y[j]);
			}
		}
		return dm;
	}
	
	public double[][] difMultipleXPartial(double [][] dm, double[][] dr, double[][] y, int offset, int runLen) {		
		for (int i = offset; i < offset + runLen; i++) {
			if (dm[i] == null) {
				dm[i] = new double[y.length];
			}			
			for (int j = 0; j < dm[i].length; j++) {
				dm[i][j] = MathUtilBase.multiple(dr[i] , y[j]);
			}
		}
		return dm;
	}
	
	public double[][] difMultipleX(final double[][] dr, final double[][] y) {
		final double [][] dx = new double[dr.length][];
		return difMultipleX(dr, y, dx);
	}
	
	@Override
	public float[][] difMultipleX(final float[][] dr, final float[][] y, final float[][] dx) {
		if (threadCnt == 1) {			
			return MathUtilBase.difMultipleX(dr, y, dx);
		}	
		
		tp.runMutipleThreads(dx.length, new PartialCallback() {			 
			public void runPartial(int offset, int runLen) {
				difMultipleXPartial(dx, dr, y, offset, runLen);	
			}
		}, threadCnt);
		return dx;
	}

	@Override
	public double[][] difMultipleX(final double[][] dr, final double[][] y, final double[][] dx) {
		if (threadCnt == 1) {			
			return MathUtilBase.difMultipleX(dr, y, dx);
		}	
		
		tp.runMutipleThreads(dx.length, new PartialCallback() {			 
			public void runPartial(int offset, int runLen) {
				difMultipleXPartial(dx, dr, y, offset, runLen);	
			}
		}, threadCnt);
		return dx;
	}

	@Override
	public double[][] difMultipleX(double[][] dr, double[] y) {
		if (threadCnt == 1) {			
			return MathUtilBase.difMultipleX(dr, y);
		}
		
		double [][] yt = transpose(new double[][]{y});
		return difMultipleX(dr, yt);
	}

	@Override
	public double[][] difMultipleX(double[] dr, double[] y) {
		if (threadCnt == 1) {			
			return MathUtilBase.difMultipleX(dr, y);
		}
		
		double [][] drt = transpose(new double[][]{dr});
		double [][] yt = transpose(new double[][]{y});
		return difMultipleX(drt, yt);
	}

	public double[] difMultipleY2v(double []dr, double [][] x) {
		if (threadCnt == 1) {			
			return MathUtilBase.difMultipleY2v(dr, x);
		}
		
		double [][] y = difMultipleY(dr, x);
		return MathUtilBase.matrix2Vector(y);
	}
	
	public double[][] difMultipleY(double []dr, double [][] x) {
		if (threadCnt == 1) {			
			return MathUtilBase.difMultipleY(dr, x);
		}
		
		double [][] drt = transpose(new double[][]{dr});
		return difMultipleY(drt, x);
	}
	
	public float[][] difMultipleYPartial(float [][] dm, float[][] dr, float[][] x, int offset, int runLen) {
		float [][] drt = transpose(dr);
		float [][] xt = transpose(x);	
		
		for (int i = offset; i < offset + runLen; i++) {
			if (dm[i] == null) {
				dm[i] = new float[dr[0].length];
			}			
			for (int j = 0; j < dm[i].length; j++) {
				dm[i][j] = MathUtilBase.multiple(drt[j] , xt[i]);
			}
		}
		return dm;
	}
	
	public double[][] difMultipleYPartial(double [][] dm, double[][] dr, double[][] x, int offset, int runLen) {
		double [][] drt = transpose(dr);
		double [][] xt = transpose(x);	
		
		for (int i = offset; i < offset + runLen; i++) {
			if (dm[i] == null) {
				dm[i] = new double[dr[0].length];
			}			
			for (int j = 0; j < dm[i].length; j++) {
				dm[i][j] = MathUtilBase.multiple(drt[j] , xt[i]);
			}
		}
		return dm;
	}
	
	public double[][] difMultipleY(final double[][] dr, final double[][] x) {
		final double [][] dm = new double[x[0].length][];
		return difMultipleY(dr, x, dm);
	}
	
	@Override
	public float[][] difMultipleY(final float[][] dr, final float[][] x, final float[][] dm) {
		if (threadCnt == 1) {			
			return MathUtilBase.difMultipleY(dr, x, dm);
		}
		
		tp.runMutipleThreads(dm.length, new PartialCallback() {			 
			public void runPartial(int offset, int runLen) {
				difMultipleYPartial(dm, dr, x, offset, runLen);	
			}
		}, threadCnt);
		return dm;
	}

	@Override
	public double[][] difMultipleY(final double[][] dr, final double[][] x, final double[][] dm) {
		if (threadCnt == 1) {			
			return MathUtilBase.difMultipleY(dr, x, dm);
		}
		
		tp.runMutipleThreads(dm.length, new PartialCallback() {			 
			public void runPartial(int offset, int runLen) {
				difMultipleYPartial(dm, dr, x, offset, runLen);	
			}
		}, threadCnt);
		return dm;
	}

	

}
