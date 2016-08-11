package deepDriver.dl.aml.linearReg;

public interface ISubject2Optimized {
	
	public int getThetasNum();
	
	public void initSubjectFunction(double [][] xVector, double [] y);
	
	public double cacluateSubject(double [] thetas);
	
	public void updateThetas(double [] thetas);

	public double getThetaDecent(int index);
	
	public double [] getInitTheta();
	
	public void setInitTheta(double [] thetas);
	
}
