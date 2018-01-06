package deepDriver.dl.aml.cnn;


public class DataMatrix implements IDataMatrix {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	double [] target;
	double result;

	double [][] matrix;	

	public double[][] getMatrix() {
		return matrix;
	}

	public void setMatrix(double[][] matrix) {
		this.matrix = matrix;
	}

	public double[] getTarget() {
		return target;
	}

	public void setTarget(double[] target) {
		this.target = target;
	}

    public double getResult() {
        return result;
    }

    public void setResult(double result) {
        this.result = result;
    } 
	
	
	
}
