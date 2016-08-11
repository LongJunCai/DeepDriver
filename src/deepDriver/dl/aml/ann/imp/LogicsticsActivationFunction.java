package deepDriver.dl.aml.ann.imp;

import java.io.Serializable;

import deepDriver.dl.aml.ann.IActivationFunction;

public class LogicsticsActivationFunction implements IActivationFunction , Serializable {

	private static final long serialVersionUID = -331809861294942272L;

	//y = 1/(1+exp(-z))
	@Override
	public double activate(double x) {
		return 1.0/(1.0+Math.exp(-x));
	}

	@Override
	public double deActivate(double x) {
		return activate(x) * (1.0 - activate(x)) ;
	}
	
	public static void main(String[] args) {
		LogicsticsActivationFunction tanhAf = new LogicsticsActivationFunction();
		System.out.println(tanhAf.activate(-1199999990));
		System.out.println(tanhAf.deActivate(1199999990));
		System.out.println(tanhAf.activate(1199999990));
		System.out.println(tanhAf.deActivate(-1190000));
	}

}
