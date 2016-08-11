package deepDriver.dl.aml.cnn;

public interface IDataMatrix {
	
	public double[][] getMatrix();

	public void setMatrix(double[][] matrix);

	public double[] getTarget();

	public void setTarget(double[] target);	
	
	public double getResult();

    public void setResult(double target); 

}
