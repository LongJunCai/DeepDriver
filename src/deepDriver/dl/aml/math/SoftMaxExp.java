package deepDriver.dl.aml.math;

public class SoftMaxExp {
	
	SigmodMatrixExp me; 
	double [] r;
	
	double beta;
	
	double [] dr;
	double [] dlr;
	
	public SoftMaxExp(int lNum, int length, double beta) {
		this.beta = beta;
		me = new SigmodMatrixExp(lNum, length);		
	}

	public void compute(double[] x) { 
		me.compute(x);
		double [] y = me.getRs();
		r = MathUtil.softMax(y, beta);
	}

	public void difCompute(double [] dy, double [] x) { 
		dr = dy;
		me.compute(x);
		dlr = MathUtil.difSoftMax4Weighting(dy, me.getRs(), beta);
		me.difCompute(dlr, x);
	}

	public double [] getR() { 
		return r;
	}
	
	public double[] getDv() { 
		return me.getDv();
	}

	public void update(double l, double m) {
		me.update(l, m);	
	}
	
	public void resetDv() {
		me.resetDv();
	}
	
}
