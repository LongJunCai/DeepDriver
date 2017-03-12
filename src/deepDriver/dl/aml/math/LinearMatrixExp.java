package deepDriver.dl.aml.math;

public class LinearMatrixExp implements IMatrixExp {
	
	LinearExp [] linearExps;

	double [] rs;
	
	double [] dr;
	
	public LinearMatrixExp(int lNum, int length) {
		super(); 
		linearExps = new LinearExp[lNum];
		for (int i = 0; i < linearExps.length; i++) {
			linearExps[i] = new LinearExp(length);
		}
		rs = new double[lNum];
	}
		
	public double[] getRs() {
		return rs;
	}

	public void setRs(double[] rs) {
		this.rs = rs;
	}

	public void compute(double [] x) {		
		for (int i = 0; i < linearExps.length; i++) {
			linearExps[i].compute(x);
			rs[i] = linearExps[i].getR();
		} 
	}
	
	public void difCompute(double [] dy, double [] x) {		
		dr = dy;
		for (int i = 0; i < linearExps.length; i++) {
			linearExps[i].difCompute(dy[i], x);  
		} 
	}
	
	public double[] getDv() { 
		double [] dv = new double[linearExps[0].getDv().length];
		for (int i = 0; i < linearExps.length; i++) {
			MathUtil.plus2V(linearExps[i].getDv(), 1.0, dv);
		} 
		return dv;
	}

	public void update(double l, double m) {
		for (int i = 0; i < linearExps.length; i++) {
			linearExps[i].update(l, m);
		} 		
	}
	
	public void resetDv() {
		for (int i = 0; i < linearExps.length; i++) {
			linearExps[i].resetDv();
		} 	 
	}

}
