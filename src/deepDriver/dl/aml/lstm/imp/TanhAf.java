package deepDriver.dl.aml.lstm.imp;

import java.io.Serializable;

import deepDriver.dl.aml.ann.IActivationFunction;

public class TanhAf implements IActivationFunction, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public double activate(double x) {
		if (x > 100) {
			return 1;
		}
		double t = Math.exp(2 * x);
		return (t - 1)/(t + 1);
	}

	@Override
	public double deActivate(double x) {
		if (x > 100) {
			return 0;
		}
		double a = activate(x);
		return 1 - a * a;
	}
	
	public static void main(String[] args) {
		TanhAf tanhAf = new TanhAf();
		System.out.println(tanhAf.activate(1190000000));
		System.out.println(tanhAf.deActivate(1190000000));
		
		System.out.println(tanhAf.activate(-1190000000));
		System.out.println(tanhAf.deActivate(-1190000000));
		System.out.println(tanhAf.activate(0));
		System.out.println(tanhAf.deActivate(0));
	}

}
