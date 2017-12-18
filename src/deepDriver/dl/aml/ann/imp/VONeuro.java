package deepDriver.dl.aml.ann.imp;

import java.io.Serializable;
import java.util.List;

import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.ann.INeuroUnit;
import deepDriver.dl.aml.ann.InputParameters;

public class VONeuro implements INeuroUnit ,Serializable {

	private static final long serialVersionUID = 2223264281984161799L;
	
	double [] results;
	public VONeuro(int inputSize) {
		results = new double[inputSize];
	}
	public void setResult(int index, double result) {
		results[index] = result;
	}
	@Override
	public double getAaz(int dataIndex) {
		return results[dataIndex];
	}

	@Override
	public double get4PropagationPreviousDelta(int dataIndex,
			int previouNeuroIndex) {
		return 0;
	}

	@Override
	public void setActivationFunction(IActivationFunction activationFunction) {

	}

	@Override
	public void forwardPropagation(List<INeuroUnit> previousNeuros,
			double[][] inputs) {

	}

	@Override
	public void backPropagation(List<INeuroUnit> previousNeuros,List<INeuroUnit> nextNeuros,
			double[][] finalResult, InputParameters parameters) {
	}
	@Override
	public void buildup(double[][] input, int position) {
		
	}
	@Override
	public void updateSelf() {		
	}
	@Override
	public double[] getThetas() {
		return null;
	}

}
