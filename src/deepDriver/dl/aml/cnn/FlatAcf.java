package deepDriver.dl.aml.cnn;

import java.io.Serializable;

import deepDriver.dl.aml.ann.IActivationFunction;

public class FlatAcf implements IActivationFunction, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public double activate(double x) {
		return x;
	}

	@Override
	public double deActivate(double x) {
		return 1;
	}

}
