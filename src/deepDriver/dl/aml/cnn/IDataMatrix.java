package deepDriver.dl.aml.cnn;

import java.io.Serializable;

public interface IDataMatrix extends Serializable {
	
	public double[][] getMatrix();

	public void setMatrix(double[][] matrix);

	public double[] getTarget();

	public void setTarget(double[] target);	
	
	public double getResult();

    public void setResult(double target); 

}
