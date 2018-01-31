package deepDriver.dl.aml.math;

import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

public class JCudaBlasMathFunction implements IMathFunction {

	@Override
	public int getThreadCnt() {
		return 0;
	}

	@Override
	public void setThreadCnt(int threadCnt) {

	}

	@Override
	public double[][] transpose(double[][] x) {
		DoubleMatrix fx = new DoubleMatrix(x);
		return fx.transpose().toArray2();
	}

	@Override
	public void minus(double[][] x, double[][] y, double[][] r) {
		DoubleMatrix fx = new DoubleMatrix(x);  
		DoubleMatrix fy = new DoubleMatrix(y);
		fy.muli(-1); 
		DoubleMatrix fr = new DoubleMatrix(r); 
		fx.addi(fy);
		fr.copy(fx); 
		double [][] ffr = fr.toArray2();
		copy2(ffr, r);
	}

	@Override
	public void multipleByElements(double[][] x, double[][] y, double[][] r) {
		DoubleMatrix fx = new DoubleMatrix(x);  
		DoubleMatrix fy = new DoubleMatrix(y); 
		DoubleMatrix fr = new DoubleMatrix(r); 
		fx.mul(fy);
		fr.copy(fx); 
		double [][] ffr = fr.toArray2();
		copy2(ffr, r);
	}

	@Override
	public void difMultipleByElements(double[][] dx, double[][] y, double[][] r) {
		multipleByElements(y, y, dx);
	}

	@Override
	public void plus(double[][] x, double xp, double[][] y, double yp, double[][] r) {
		DoubleMatrix fx = new DoubleMatrix(x);
		fx.muli(xp); 
		DoubleMatrix fy = new DoubleMatrix(y);
		fy.muli(yp);
		DoubleMatrix fr = new DoubleMatrix(r); 
		fx.addi(fy);
		fr.copy(fx);
		double [][] ffr = fr.toArray2();
		copy2(ffr, r);
	}

	@Override
	public void plus(float[][] x, float xp, float[][] y, float yp, float[][] r) {
		FloatMatrix fx = new FloatMatrix(x);
		fx.muli(xp); 
//		System.out.println(fx);
		FloatMatrix fy = new FloatMatrix(y);
		fy.muli(yp);
//		System.out.println(fy);
		FloatMatrix fr = new FloatMatrix(r); 
		fx.addi(fy);
		fr.copy(fx);
//		System.out.println(fr);
		float [][] ffr = fr.toArray2();
		copy2(ffr, r);
	}

	@Override
	public void plus(double[][] x, double[][] y, double yp, double[][] r) {
		plus(x, 1, y, yp, r);
	}

	@Override
	public void plus(float[][] x, float[][] y, float yp, float[][] r) {
		FloatMatrix fx = new FloatMatrix(x); 
		FloatMatrix fy = new FloatMatrix(y);
		fy.muli(yp); 
		fx.addi(fy);
		float [][] d = fx.toArray2();
		copy2(d, r);
	}
	
	public void copy2(float[][] x, float[][] y) {
		for (int i = 0; i < x.length; i++) {
			for (int j = 0; j < x[i].length; j++) {
				y[i][j] = x[i][j];
			}
		}
	}
	
	public void copy2(double [][] x, double[][] y) {
		for (int i = 0; i < x.length; i++) {
			for (int j = 0; j < x[i].length; j++) {
				y[i][j] = x[i][j];
			}
		}
	}


	@Override
	public void set(double[][] x, double[][] y) {
		copy2(x, y);
	}

	@Override
	public double[] multipleV2v(double[][] x, double[] y) {
		DoubleMatrix fm = new DoubleMatrix(x);
		DoubleMatrix fm1 = new DoubleMatrix(y);
		DoubleMatrix fm2 = fm.mmul(fm1);  
		return fm2.toArray();
	}

	@Override
	public double[][] multipleV(double[][] x, double[] y) {
		DoubleMatrix fm = new DoubleMatrix(x);
		DoubleMatrix fm1 = new DoubleMatrix(y);
		DoubleMatrix fm2 = fm.mmul(fm1);  
		return fm2.toArray2(); 
	}

	@Override
	public double[][] multiple(double[][] x, double[][] y) {
		DoubleMatrix fm = new DoubleMatrix(x);
		DoubleMatrix fm1 = new DoubleMatrix(y);
		DoubleMatrix fm2 = fm.mmul(fm1);  
		return fm2.toArray2();
	}

	@Override
	public double[][] multiple(double[][] x, double[][] y, double[][] r) {
		DoubleMatrix fm = new DoubleMatrix(x);
		DoubleMatrix fm1 = new DoubleMatrix(y);
		DoubleMatrix fm2 = fm.mmul(fm1); 
		copy2(fm2.toArray2(), r);
		return r;
	}

	@Override
	public float[][] multiple(float[][] x, float[][] y, float[][] r) {
		FloatMatrix fm = new FloatMatrix(x);
		FloatMatrix fm1 = new FloatMatrix(y);
		FloatMatrix fm2 = fm.mmul(fm1);
//		FloatMatrix fm3 = new FloatMatrix(r);
//		fm3.copy(fm2);
		copy2(fm2.toArray2(), r);
		return r;
	}

	@Override
	public double[][] difMultipleX(double[][] dr, double[][] y) {
		DoubleMatrix fm = new DoubleMatrix(dr);
		DoubleMatrix fm1 = new DoubleMatrix(y);
		DoubleMatrix fm2 = fm.mmul(fm1.transpose()); 
		return fm2.toArray2();
	}

	@Override
	public double[][] difMultipleX(double[][] dr, double[][] y, double[][] dx) {
		DoubleMatrix fm = new DoubleMatrix(dr);
		DoubleMatrix fm1 = new DoubleMatrix(y);
		DoubleMatrix fm2 = fm.mmul(fm1.transpose());
		copy2(fm2.toArray2(), dx);
		return dx;
	}

	@Override
	public float[][] difMultipleX(float[][] dr, float[][] y, float[][] dx) {
		FloatMatrix fm = new FloatMatrix(dr);
		FloatMatrix fm1 = new FloatMatrix(y);
		FloatMatrix fm2 = fm.mmul(fm1.transpose());
//		FloatMatrix fm3 = new FloatMatrix(dx);
		copy2(fm2.toArray2(), dx);
		return dx;
	}

	@Override
	public double[][] difMultipleX(double[][] dr, double[] y) { 
		DoubleMatrix fm = new DoubleMatrix(dr);
		DoubleMatrix fm1 = new DoubleMatrix(y);
		DoubleMatrix fm2 = fm.mmul(fm1.transpose()); 
		return fm2.toArray2();
	}

	@Override
	public double[][] difMultipleX(double[] dr, double[] y) {
		DoubleMatrix fm = new DoubleMatrix(dr);
		DoubleMatrix fm1 = new DoubleMatrix(y);
		DoubleMatrix fm2 = fm.mmul(fm1.transpose()); 
		return fm2.toArray2();
	}

	@Override
	public double[] difMultipleY2v(double[] dr, double[][] x) {
		DoubleMatrix fm = new DoubleMatrix(dr);
		DoubleMatrix fm1 = new DoubleMatrix(x);
		DoubleMatrix fm2 = fm1.transpose().mmul(fm); 
		return fm2.toArray();
	}

	@Override
	public double[][] difMultipleY(double[] dr, double[][] x) {
		DoubleMatrix fm = new DoubleMatrix(dr);
		DoubleMatrix fm1 = new DoubleMatrix(x);
		DoubleMatrix fm2 = fm1.transpose().mmul(fm); 
		return fm2.toArray2();
	}

	@Override
	public double[][] difMultipleY(double[][] dr, double[][] x) {
		DoubleMatrix fm = new DoubleMatrix(dr);
		DoubleMatrix fm1 = new DoubleMatrix(x);
		DoubleMatrix fm2 = fm1.transpose().mmul(fm); 
		return fm2.toArray2();
	}

	@Override
	public double[][] difMultipleY(double[][] dr, double[][] x, double[][] dy) {
		DoubleMatrix fm = new DoubleMatrix(dr);
		DoubleMatrix fm1 = new DoubleMatrix(x);
		DoubleMatrix fm2 = fm1.transpose().mmul(fm); 
		copy2(fm2.toArray2(), dy);
		return dy;
	}

	@Override
	public float[][] difMultipleY(float[][] dr, float[][] x, float[][] dy) {
		FloatMatrix fm = new FloatMatrix(dr);
		FloatMatrix fm1 = new FloatMatrix(x);
		FloatMatrix fm2 = fm1.transpose().mmul(fm);
//		FloatMatrix fm3 = new FloatMatrix(dy);
//		fm3.copy(fm2);
		copy2(fm2.toArray2(), dy);
		return dy;
	}

}
