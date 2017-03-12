package deepDriver.dl.aml.math;

public class ContentBasedWeighting {
	
	double [][] matrix; 
	double [][] dm; 
	
	public double[][] getMatrix() {
		return matrix;
	}
	
	public void setMatrix(double[][] matrix) {
		this.matrix = matrix;
	}
	
	double [] sims;
	double [] k;
	double beta;
	double [] dk;
	
	double [] sm;
	
	public double[] getSm() {
		return sm;
	}

	public void setSm(double[] sm) {
		this.sm = sm;
	}

	public double [] weighting(double [] k, double beta) {
		this.k = k;
		this.beta = beta;
		if (sims == null) {
			sims = new double[matrix.length];
		}
		for (int i = 0; i < sims.length; i++) {
			sims[i] = MathUtil.cos(k, matrix[i]);			
		}
		sm = MathUtil.softMax(sims, beta);
		return sm;
	}
	
	double dbeta = 0;
	
	public double [] backWeighting(double [] da, double [] k, double beta) {
		weighting(k, beta);
		
		dm = MathUtil.allocate(matrix.length, matrix[0].length);
		double [] dsims = MathUtil.difSoftMax4Weighting(da, sims, beta);
		dbeta = MathUtil.difSoftMax4Beta(da, sims, beta);
		dk = new double[k.length];
		
		for (int i = 0; i < matrix.length; i++) {
			double [] dk1 = new double[k.length];
			MathUtil.difCos(dsims[i], dk1, k, matrix[i]);
			MathUtil.plus2V(dk1, dk);
			
			MathUtil.difCos(dsims[i], dm[i], matrix[i], k);
		}
		return dk;
	}
	
	public void update(double l, double m) {
		
	}

	public double[][] getDm() {
		return dm;
	} 

	public double[] getDk() {
		return dk;
	}

	public double getDbeta() {
		return dbeta;
	}
}
