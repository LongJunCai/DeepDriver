package deepDriver.dl.aml.linearReg;

public class LinearFunctionSubject implements ISubject2Optimized {
	
	double[] thetas;
	double[][] xVector;
	double[] y;
	@Override
	public int getThetasNum() {
//		if (xVector == null) {
//			throw new Exception();
//		}
		return xVector[0].length;
	}

	@Override
	public void initSubjectFunction(double[][] xVector, double[] y) {
		this.xVector = xVector;
		this.y = y;
		this.thetas = new double[xVector[0].length];
	}

	@Override
	public double cacluateSubject(double[] thetas) {
		double sum = 0; 
		for (int i = 0; i < y.length; i++) {
			double yi = y[i];
			double [] x = xVector[i];
			double fxi = 0;
			for (int j = 0; j < x.length; j++) {
				fxi = fxi + thetas[j] * x[j];
			}
			double residual = (yi - fxi);
			sum = sum + ( residual * residual);
		}
//		return sum/(2.0 * (double)(y.length));
		return sum/(2.0);
	}

	@Override
	public void updateThetas(double[] thetas) {
		for (int i = 0; i < thetas.length; i++) {
			this.thetas[i] = thetas[i];
		}		
	}

	@Override
	public double getThetaDecent(int index) {
		//(f(x) - y)* xi
		double gradient = 0;
		for (int i = 0; i < y.length; i++) {
			double yi = y[i];
			double [] x = xVector[i];
			double fx = 0;
			for (int j = 0; j < x.length; j++) {
				fx = fx + thetas[j] * x[j];
			}
			double residual = (fx - yi);
			gradient = gradient + residual * x[index];
		}
//		return gradient/(double)(y.length);
		return gradient;
	}
	
	public static void main(String[] args) {
		double [][] xVector = {{1, 2,3},{4,5,6}};
		System.out.println(xVector.length);
	}

	double [] initThetas;
	@Override
	public double[] getInitTheta() {
		return initThetas;
	}

	@Override
	public void setInitTheta(double[] thetas) {
		this.initThetas = thetas;
	}

}
