package deepDriver.dl.aml.linearReg;

import deepDriver.dl.aml.cart.DataSet;

public class LinearRegression {
	GradientDecentOptimizer gradientDecentOptimizer = 
			new GradientDecentOptimizer();
	
	public LinearExpression fit(DataSet ds) {
		LinearFunctionSubject linearFunctionSubject = new LinearFunctionSubject();
		double [] thetas = gradientDecentOptimizer.
				optimizeFunction(linearFunctionSubject, ds.getDependentVars(), ds.getIndependentVars(), false);
		return new LinearExpression(thetas);
	}

}
