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

	public double[][] transposePartial(double [][] t, double[][] x, int offset, int runLen) { 		
		for (int i = offset; i < offset + runLen; i++) {
			t[i] = new double[x.length];
			for (int j = 0; j < t[i].length; j++) {
				t[i][j] = x[j][i];
			}
		}
		return t;
	}

	@Override
	public double[][] transpose(double[][] x) {
		if (threadCnt == 1) {
			return MathUtilBase.transpose(x);
		}
		
		double [][] t = new double[x[0].length][];
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
	public void minus(double[][] x, double[][] y, double[][] r) {
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
	public void multipleByElements(double[][] x, double[][] y, double[][] r) {
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
	public void difMultipleByElements(double[][] dx, double[][] y, double[][] r) {
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
	
	public void plusPartial(double[][] x, double xp, double[][] y, double yp, double[][] r, int offset, int runLen) {
		for (int i = offset; i < offset + runLen; i++) { 
			for (int j = 0; j < x[i].length; j++) {
				r[i][j] = xp * x[i][j] + yp * y[i][j];
			}
		} 
	}

	@Override
	public void plus(double[][] x, double xp, double[][] y, double yp, double[][] r) {
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
	public void set(double[][] x, double[][] y) {
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
	public double[][] multipleV(double[][] x, double[] y) {
		if (threadCnt == 1) {			
			return MathUtilBase.multipleV(x, y);
		}
		
		double[][] result = new double[x.length][];
		tp.runMutipleThreads(result.length, new PartialCallback() {			 
			public void runPartial(int offset, int runLen) {
				multipleVPartial(result, x, y, offset, runLen);	
			}
		}, threadCnt);		
		return result;
	}
	
	public double[][] multiplePartial(double[][] result, double[][] x, double[][] y, int offset, int runLen) {
		double [][] t = transpose(y);		
		for (int i = offset; i < offset + runLen; i++) {
			result[i] = new double[t.length];
			for (int j = 0; j < result[i].length; j++) {
				result[i][j] = MathUtilBase.multiple(x[i], t[j]);
			}
		}
		return result;
	}

	@Override
	public double[][] multiple(double[][] x, double[][] y) {
		if (threadCnt == 1) {			
			return MathUtilBase.multiple(x, y);
		}
		
		double[][] result = new double[x.length][];
		tp.runMutipleThreads(result.length, new PartialCallback() {			 
			public void runPartial(int offset, int runLen) {
				multiplePartial(result, x, y, offset, runLen);	
			}
		}, threadCnt);
		return result;
	}
	
	public double[][] difMultipleXPartial(double [][] dm, double[][] dr, double[][] y, int offset, int runLen) {		
		for (int i = offset; i < offset + runLen; i++) {
			dm[i] = new double[y.length];
			for (int j = 0; j < dm[i].length; j++) {
				dm[i][j] = MathUtilBase.multiple(dr[i] , y[j]);
			}
		}
		return dm;
	}

	@Override
	public double[][] difMultipleX(double[][] dr, double[][] y) {
		if (threadCnt == 1) {			
			return MathUtilBase.difMultipleX(dr, y);
		}
		
		double [][] dm = new double[dr.length][];
		tp.runMutipleThreads(dm.length, new PartialCallback() {			 
			public void runPartial(int offset, int runLen) {
				difMultipleXPartial(dm, dr, y, offset, runLen);	
			}
		}, threadCnt);
		return dm;
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
	
	public double[][] difMultipleYPartial(double [][] dm, double[][] dr, double[][] x, int offset, int runLen) {
		double [][] drt = transpose(dr);
		double [][] xt = transpose(x);	
		
		for (int i = offset; i < offset + runLen; i++) {
			dm[i] = new double[dr[0].length];
			for (int j = 0; j < dm[i].length; j++) {
				dm[i][j] = MathUtilBase.multiple(drt[j] , xt[i]);
			}
		}
		return dm;
	}

	@Override
	public double[][] difMultipleY(double[][] dr, double[][] x) {
		if (threadCnt == 1) {			
			return MathUtilBase.difMultipleY(dr, x);
		}
		
		double [][] dm = new double[x[0].length][];
		tp.runMutipleThreads(dm.length, new PartialCallback() {			 
			public void runPartial(int offset, int runLen) {
				difMultipleYPartial(dm, dr, x, offset, runLen);	
			}
		}, threadCnt);
		return dm;
	}

	

}
