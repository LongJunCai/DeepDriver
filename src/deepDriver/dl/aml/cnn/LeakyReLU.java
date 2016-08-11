package deepDriver.dl.aml.cnn;

import java.io.Serializable;

import deepDriver.dl.aml.ann.IActivationFunction;

public class LeakyReLU implements IActivationFunction, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	double a = 0.001;
	

	public double getA() {
		return a;
	}

	public void setA(double a) {
		this.a = a;
	}

	@Override
	public double activate(double x) {
		if (x < 0) {
			return x * a;
		}
		return x;
	}

	@Override
	public double deActivate(double x) {
		if (x < 0) {
			return a;
		}
		return 1;
	}

}