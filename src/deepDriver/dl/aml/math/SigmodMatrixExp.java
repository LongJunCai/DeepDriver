package deepDriver.dl.aml.math;

public class SigmodMatrixExp implements IMatrixExp {
	
	SigmodExp [] exps;

	double [] rs;
	
	double [] dr;
	
	public SigmodMatrixExp(int lNum, int length) {
		super(); 
		exps = new SigmodExp[lNum];
		for (int i = 0; i < exps.length; i++) {
			exps[i] = new SigmodExp(length);
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
		for (int i = 0; i < exps.length; i++) {
			exps[i].compute(x);
			rs[i] = exps[i].getR();
		} 
	}
	
	public void difCompute(double [] dy, double [] x) {		
		dr = dy;
		for (int i = 0; i < exps.length; i++) {
			exps[i].difCompute(dy[i], x);  
		} 
	}
	
	public double[] getDv() { 
		double [] dv = new double[exps[0].getDv().length];
		for (int i = 0; i < exps.length; i++) {
			MathUtil.plus2V(exps[i].getDv(), 1.0, dv);
		} 
		return dv;
	}

	public void update(double l, double m) {
		for (int i = 0; i < exps.length; i++) {
			exps[i].update(l, m);
		} 		
	}
	
	public void resetDv() {
		for (int i = 0; i < exps.length; i++) {
			exps[i].resetDv();
		} 	 
	}

}
