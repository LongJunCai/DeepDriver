package deepDriver.dl.aml.cart;

import java.io.Serializable;

public class DataSet implements Serializable {
	double [][] dependentVars;
	String [] labels;
	double [] independentVars;
	
	public double[][] getDependentVars() {
		return dependentVars;
	}
	public void setDependentVars(double[][] dependentVars) {
		this.dependentVars = dependentVars;
	}
	public String[] getLabels() {
		return labels;
	}
	public void setLabels(String[] labels) {
		this.labels = labels;
	}
	public double[] getIndependentVars() {
		return independentVars;
	}
	public void setIndependentVars(double[] independentVars) {
		this.independentVars = independentVars;
	}
	
}
