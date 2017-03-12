package deepDriver.dl.aml.math;

public class SigmodExp implements IExp4Function {
	
	double r;
	LinearExp le;
	
	double dr;
	double dlr;
	
	
	public SigmodExp(int length) {
		this.le = new LinearExp(length); 
	}
	
	public void compute(double [] x) {
		le.compute(x);
		r = MathUtil.sigmod(le.getR()); 
	}
	
	public double getR() {
		return r;
	}

	@Override
	public void difCompute(double dy, double [] x) { 
		le.compute(x);
		dr = dy;
		dlr = MathUtil.difSigmod(le.getR()) * dy;
		le.difCompute(dlr, x);
	}
	
	public double[] getDv() { 
		return le.getDv();
	}

	@Override
	public void update(double l, double m) {
		le.update(l, m);		
	}

	@Override
	public void resetDv() {
		le.resetDv();
	}

	public double[] getX() { 
		return le.x;
	}
	
	public double[] getPara() { 
		return le.parameters;
	}
	
	public double[] getDl() { 
		return le.deltaPara;
	}
	
	public double[] getDl2() { 
		return le.deltaPara2;
	}
}
