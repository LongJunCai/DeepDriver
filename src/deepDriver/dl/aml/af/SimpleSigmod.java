package deepDriver.dl.aml.af;

import java.io.Serializable;

import deepDriver.dl.aml.ann.IActivationFunction;

public class SimpleSigmod implements IActivationFunction, Serializable {

 
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	//y = 1/(1+exp(-z))
	@Override
	public double activate(double x) {
		if (x >= 8) {
			return 1;
		} 
		if (x <= -8) {
			return 0;
		}
		return 1.0/(1.0+Math.exp(-x));
	}

	@Override
	public double deActivate(double x) {
		if (x <= -8 || x >= 8) {
			return 0;
		}
		return activate(x) * (1.0 - activate(x)) ;
	}
	
	public static void main(String[] args) {
		SimpleSigmod tanhAf = new SimpleSigmod();
		System.out.println(tanhAf.activate(-8)); 		
		System.out.println(tanhAf.activate(8));
		
		System.out.println(tanhAf.deActivate(-8));
		System.out.println(tanhAf.deActivate(8));
		System.out.println(tanhAf.deActivate(0.5));
	}

}
