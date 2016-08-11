package deepDriver.dl.aml.cnn;

import java.io.Serializable;

import deepDriver.dl.aml.ann.IActivationFunction;

public class ReLU implements IActivationFunction, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public double activate(double x) {
		if (x < 0) {
			return 0;
		}
		return x;
	}

	@Override
	public double deActivate(double x) {
		if (x < 0) {
			return 0;
		}
		return 1;
	}

}
