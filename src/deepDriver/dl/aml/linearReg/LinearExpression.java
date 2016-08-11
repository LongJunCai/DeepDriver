package deepDriver.dl.aml.linearReg;

import java.io.Serializable;

import deepDriver.dl.aml.cart.DataSet;

public class LinearExpression implements Serializable {
	
	private static final long serialVersionUID = 1L;
	double [] thetas;
	public LinearExpression(double[] thetas) {
		super();
		this.thetas = thetas;
	}
	
	public double[] predict(DataSet ds) {
		double [][] vars = ds.getDependentVars();
		double [] ys = new double[vars.length];
		for (int i = 0; i < ys.length; i++) {
			for (int j = 0; j < thetas.length; j++) {
				ys[i] = ys[i] + thetas[j] * vars[i][j];
			}
		}
		return ys;
	}	
	
	public double[] getThetas() {
		return thetas;
	}
	

}
