package deepDriver.dl.aml.math;
 
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;

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
	
	public float [][] transpose(float [][] x) {
		FloatMatrix fx = new FloatMatrix(x);
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
		JCublas.cublasInit();
		Pointer p_x = new Pointer();
		Pointer p_y = new Pointer();
		Pointer p_r = new Pointer();
		double [] xd = matrix2Arr(x);
		double [] yd = matrix2Arr(y);
		double [] rd = new double[x.length * y[0].length];
		JCublas.cublasAlloc(xd.length, Sizeof.DOUBLE, p_x);
		JCublas.cublasAlloc(yd.length, Sizeof.DOUBLE, p_y);
		JCublas.cublasAlloc(rd.length, Sizeof.DOUBLE, p_r);
		
		// Copy the memory from the host to the device
        JCublas.cublasSetVector(xd.length, Sizeof.DOUBLE, Pointer.to(xd), 1, p_x, 1);
        JCublas.cublasSetVector(yd.length, Sizeof.DOUBLE, Pointer.to(yd), 1, p_y, 1);
        JCublas.cublasSetVector(rd.length, Sizeof.DOUBLE, Pointer.to(rd), 1, p_r, 1);
        int pyRow = y[0].length;
        int pxColumn = x.length;
        int pyColumn = y.length; 
        int pxRow = x[0].length;

        // Execute sgemm
        JCublas.cublasSgemm(
            'n', 'n', pyRow, pxColumn, pyColumn, 1.0f, p_y, pyRow, p_x, pxRow, 1.0f, p_r, pyRow);

        // Copy the result from the device to the host
        JCublas.cublasGetVector(rd.length, Sizeof.DOUBLE, p_r, 1, Pointer.to(rd), 1);

        // Clean up
        JCublas.cublasFree(p_x);
        JCublas.cublasFree(p_y);
        JCublas.cublasFree(p_r);
		
		JCublas.cublasShutdown();
		return arr2Matrix(x.length, y[0].length, rd);
	}
	
	public double [] matrix2Arr(double[][] x) {
		double [] a = new double[x.length * x[0].length];
		int cnt = 0;
		for (int i = 0; i < x[0].length; i++) {
			for (int j = 0; j < x.length; j++) {
				a[cnt ++] = x[j][i];
			}
		}
		return a;
	}
	
	public float [] matrix2Arr(float [][] x) {
		float [] a = new float[x.length * x[0].length];
		int cnt = 0;
		for (int i = 0; i < x[0].length; i++) {
			for (int j = 0; j < x.length; j++) {
				a[cnt ++] = x[j][i];
			}
		}
		return a;
	}
	
	public float [][] arr2Matrix(int r, int c, float [] fx) {
		float [][] x = new float[r][c];
		int cnt = 0;
		for (int i = 0; i < x[0].length; i++) {
			for (int j = 0; j < x.length; j++) {
				 x[j][i] = fx[cnt ++];
			}
		}
		return x;
	}
	
	public double [][] arr2Matrix(int r, int c, double [] fx) {
		double [][] x = new double[r][c];
		int cnt = 0;
		for (int i = 0; i < x[0].length; i++) {
			for (int j = 0; j < x.length; j++) {
				 x[j][i] = fx[cnt ++];
			}
		}
		return x;
	}

	@Override
	public double[][] multiple(double[][] x, double[][] y, double[][] r) {
		double [][] rx = multiple(x, y);
		copy2(rx, r);
		return r;
	}

	@Override
	public float[][] multiple(float[][] x, float[][] y, float[][] r) {
		JCublas.cublasInit();
		Pointer p_x = new Pointer();
		Pointer p_y = new Pointer();
		Pointer p_r = new Pointer();
		float [] xd = matrix2Arr(x);
		float [] yd = matrix2Arr(y);
		float [] rd = new float[x.length * y[0].length];
		JCublas.cublasAlloc(xd.length, Sizeof.FLOAT, p_x);
		JCublas.cublasAlloc(yd.length, Sizeof.FLOAT, p_y);
		JCublas.cublasAlloc(rd.length, Sizeof.FLOAT, p_r);
		
		// Copy the memory from the host to the device
        JCublas.cublasSetVector(xd.length, Sizeof.FLOAT, Pointer.to(xd), 1, p_x, 1);
        JCublas.cublasSetVector(yd.length, Sizeof.FLOAT, Pointer.to(yd), 1, p_y, 1);
        JCublas.cublasSetVector(rd.length, Sizeof.FLOAT, Pointer.to(rd), 1, p_r, 1);
        int pyRow = y[0].length;
        int pxColumn = x.length;
        int pyColumn = y.length; 
        int pxRow = x[0].length;

        // Execute sgemm
        JCublas.cublasSgemm(
            'n', 'n', pyRow, pxColumn, pyColumn, 1.0f, p_y, pyRow, p_x, pxRow, 1.0f, p_r, pyRow);

        // Copy the result from the device to the host
        JCublas.cublasGetVector(rd.length, Sizeof.FLOAT, p_r, 1, Pointer.to(rd), 1);

        // Clean up
        JCublas.cublasFree(p_x);
        JCublas.cublasFree(p_y);
        JCublas.cublasFree(p_r);
		
		JCublas.cublasShutdown();
		float [][] fx = arr2Matrix(x.length, y[0].length, rd); 
		copy2(fx, r);
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
//		FloatMatrix fm = new FloatMatrix(dr);
//		FloatMatrix fm1 = new FloatMatrix(y);
//		FloatMatrix fm2 = fm.mmul(fm1.transpose());
		multiple(dr, transpose(y), dx);
//		FloatMatrix fm3 = new FloatMatrix(dx); 
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
//		FloatMatrix fm = new FloatMatrix(dr);
//		FloatMatrix fm1 = new FloatMatrix(x);
//		FloatMatrix fm2 = fm1.transpose().mmul(fm);
//		FloatMatrix fm3 = new FloatMatrix(dy);
//		fm3.copy(fm2);
		multiple(transpose(x), dr, dy);
//		copy2(fm2.toArray2(), dy);
		return dy;
	}

}
