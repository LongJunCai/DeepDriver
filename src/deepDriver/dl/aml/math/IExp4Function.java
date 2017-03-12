package deepDriver.dl.aml.math;

public interface IExp4Function {
	/*
	 * y = f(x), x is a vector
	 * **/
	public void compute(double [] x);
	
	public void difCompute(double dy, double [] x);
	
	public double getR();
	
	public double [] getDv();
	
	public void resetDv();
	
	public void update(double l, double m);

}
