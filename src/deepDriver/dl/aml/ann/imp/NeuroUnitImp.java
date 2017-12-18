package deepDriver.dl.aml.ann.imp;

import java.io.Serializable;
import java.util.List;
import java.util.Random;

import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.ann.INeuroUnit;
import deepDriver.dl.aml.ann.InputParameters;

public class NeuroUnitImp implements INeuroUnit , Serializable {

	private static final long serialVersionUID = -9221602989423182893L;
	
	public static Random random = new Random(System.currentTimeMillis());
	protected double min = 0;
	protected double max = 1.0;
	protected double length = max - min;
	protected IActivationFunction activationFunction;
	protected double [] aas;
	protected double [] zzs;
	protected double [] deltaZ;
	
	protected double [] thetas;	
	protected double [] deltaThetas;
	
	
	public double[] getThetas() {
		return thetas;
	}

//	public void setThetas(double[] thetas) {
//		this.thetas = thetas;
//	}

	@Override
	public double get4PropagationPreviousDelta(int dataIndex,
			int previouNeuroIndex) {		
		return deltaZ[dataIndex] * thetas[previouNeuroIndex];
	}

	@Override
	public void setActivationFunction(IActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
	}
	
	protected boolean randomize = true;
	
	protected void initTheta() {
		if (randomize) {
			for (int i = 0; i < thetas.length; i++) {
				thetas[i] = length * random.nextDouble()
					+ min;
			}
		}		
	}

	@Override
	public void forwardPropagation(List<INeuroUnit> previousNeuros, double [][] input) {
		if (thetas == null) {
			this.thetas = new double[previousNeuros.size() + 1];
			initTheta();
		}		
		this.aas = new double[input.length];
		zzs = new double[input.length];
		for (int i = 0; i < aas.length; i++) {			
			double z = 0;
			for (int j = 0; j < previousNeuros.size(); j++) {
				z = z + thetas[j] * previousNeuros.get(j).getAaz(i);
			}
			z = z + thetas[thetas.length - 1];
			zzs[i] = z;
			double a = activationFunction.activate(z);
			aas[i] = a;	
		}
	}

	@Override
	public void backPropagation(List<INeuroUnit> previousNeuros, List<INeuroUnit> nextNeuros, double [][] result, InputParameters parameters) {
		if (deltaZ == null) {			
			deltaThetas = new double[thetas.length];
		}
		deltaZ = new double[aas.length];
		if (nextNeuros == null) {			
			for (int i = 0; i < deltaZ.length; i++) {
				deltaZ[i] = (aas[i] - result[i][position]) * activationFunction.deActivate(zzs[i]);
			}
		} else {
			for (int i = 0; i < deltaZ.length; i++) {
				double sumDelta = 0;
				for (int j = 0; j < nextNeuros.size(); j++) {
					sumDelta = sumDelta + nextNeuros.get(j).get4PropagationPreviousDelta(i, position);
				}
				deltaZ[i] = (sumDelta) * activationFunction.deActivate(zzs[i]);
			}
		}
		if (thetas == null) {
			return;
		}
		for (int i = 0; i < thetas.length; i++) {
			double delta4theta = 0;
			if (i < thetas.length - 1) {
				for (int j = 0; j < deltaZ.length; j++) {
					delta4theta = delta4theta + 
						deltaZ[j] * previousNeuros.get(i).getAaz(j);
				}
			} else {
				for (int j = 0; j < deltaZ.length; j++) {
					delta4theta = delta4theta + 
						deltaZ[j];
				}
			}
			deltaThetas[i] = - parameters.getAlpha() 
					* delta4theta;
			setAlpha(parameters.getAlpha());
			this.lamda = parameters.getLamda();
		}
	}
	
	protected double alpha;
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}
	
	public double getAlpha() {
		return alpha;
	}	

	public double[] getZzs() {
		return zzs;
	}

	public void setZzs(double[] zzs) {
		this.zzs = zzs;
	}

	@Override
	public double getAaz(int dataIndex) {
		return aas[dataIndex];
	}	

	public double[] getAas() {
		return aas;
	}

	public void setAas(double[] aas) {
		this.aas = aas;
	}	

	public double[] getDeltaZ() {
		return deltaZ;
	}

	public void setDeltaZ(double[] deltaZ) {
		this.deltaZ = deltaZ;
	}

	protected int position;
	@Override
	public void buildup(double[][] input, int position) {
		this.position = position;
	}
	
	protected double lamda = 0.00001;
	protected double momentum = 0.9;
	protected double [] lastDeltaThetas;

	@Override
	public void updateSelf() {
		if (thetas == null) {
			return;
		}
		if (lastDeltaThetas == null) {
			lastDeltaThetas = new double [deltaThetas.length];
		}
		double deltaW = 0;
		for (int i = 0; i < thetas.length; i++) {
			//no regularization 
			//thetas[i] = thetas[i] + deltaThetas[i];
			/**
			 * Add regularization to avoid overfitting
			 * **/
			/**
			 * Add momentum to accelerate
			 * */
			deltaW = deltaThetas[i] + 
					momentum * lastDeltaThetas[i];
			/**
			 * Add momentum to accelerate
			 * */
			if (i == thetas.length - 1) {
				thetas[i] = thetas[i] + deltaW; //deltaThetas[i];
			} else {
				thetas[i] = thetas[i] + deltaW //deltaThetas[i]
					- getAlpha()* lamda*  thetas[i] ;
			}
			lastDeltaThetas[i] = deltaW;
			/**
			 * Add regularization to avoid overfitting
			 * **/
		}		
	}


}
