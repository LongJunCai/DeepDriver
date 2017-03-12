package deepDriver.dl.aml.math;

public class OnePlusExp implements IExp4Function {
	
	LinearExp le; 
	double r;
	
	double dr;
	double dlr;
	
	public OnePlusExp(int length) {
		this.le = new LinearExp(length); 
	}
	
	public void compute(double [] x) {
		le.compute(x);
		r = MathUtil.onePlus(le.getR());
	}
	
	public double [] getX() {
		return le.x;
	}
	
	public double getR() {
		return r;
	}

	@Override
	public void difCompute(double dy, double [] x) { 
		dr = dy;
		le.compute(x);
		dlr = MathUtil.difOnePlus(le.getR()) * dy;
		le.difCompute(dlr, x);
	}

	@Override
	public double[] getDv() { 
		return le.getDv();
	}

	@Override
	public void update(double l, double m) {
		le.update(l, m);		
	}
	
	public void resetDv() {
		le.resetDv();
	}
	
	public static void main(String[] args) {
		OnePlusExp one = new OnePlusExp(6);
		double [] x = {0,0,0,0,0,0};
		one.compute(x);
		System.out.println(one.getR());
		System.out.println(MathUtil.onePlus(0));
		
	}

}
