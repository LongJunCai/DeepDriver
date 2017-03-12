package deepDriver.dl.aml.math;

public interface IMatrixExp {
	
	public double[] getRs();

	public void setRs(double[] rs);

	public void compute(double [] x);
	
	public void difCompute(double [] dy, double [] x);
	
	public double[] getDv();

	public void update(double l, double m);
	
	public void resetDv();

}
