package deepDriver.dl.aml.ann.imp;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.ann.ILayer;
import deepDriver.dl.aml.ann.INeuroUnit;
import deepDriver.dl.aml.ann.InputParameters;

public class LayerImp implements ILayer , Serializable {
	
	private static final long serialVersionUID = 2027416121480770704L;
	
	int pos;
	ILayer nextLayer;
	ILayer previousLayer;
	List<INeuroUnit> neuros = new ArrayList<INeuroUnit>();
	
	public int getPos() {
		return pos;
	}

	public void setPos(int pos) {
		this.pos = pos;
	}
	
	public ILayer getNextLayer() {
		return nextLayer;
	}

	public void setNextLayer(ILayer nextLayer) {
		this.nextLayer = nextLayer;
	}

	public ILayer getPreviousLayer() {
		return previousLayer;
	}

	public void setPreviousLayer(ILayer previousLayer) {
		this.previousLayer = previousLayer;
	}

	@Override
	public void addNeuro(INeuroUnit neuro) {
		neuros.add(neuro);
	}

	@Override
	public List<INeuroUnit> getNeuros() {
		return neuros;
	}

	@Override
	public void forwardPropagation(double [][] input) {
		if (getPreviousLayer() == null) {
			updateValues4FirstLayer(input);
			return;
		}
		for (int i = 0; i < neuros.size(); i++) {
			INeuroUnit neuro = neuros.get(i);
			neuro.forwardPropagation(this.getPreviousLayer().getNeuros(), input);
		}
	}

	@Override
	public void backPropagation(double [][] finalResult, InputParameters parameters) {
		if (getPreviousLayer() == null) {
			return;
		}
		for (int i = 0; i < neuros.size(); i++) {
			INeuroUnit neuro = neuros.get(i);
			neuro.backPropagation(this.getPreviousLayer().getNeuros(), getNextLayer() == null? null : this.getNextLayer().getNeuros(), finalResult, parameters);
		}
	}
	
	public void updateValues4FirstLayer(double[][] input) {
		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[0].length; j++) {
				if (j > neuros.size() -1) {
					VONeuro vONeuro = new VONeuro(input.length);
					this.neuros.add(vONeuro);
				} 
				VONeuro vONeuro = (VONeuro) neuros.get(j);
				vONeuro.setResult(i, input[i][j]);
			}
		}
	}

	@Override
	public void buildup(ILayer previousLayer, double[][] input, IActivationFunction acf, boolean isLastLayer, int neuroCount) {
		this.previousLayer = previousLayer;
		if (previousLayer != null) {
			previousLayer.setNextLayer(this);
		}
//		if (isLastLayer) {
//			NeuroUnitImp neuroUnitImp = new NeuroUnitImp();
//			neuroUnitImp.buildup(input, 0);
//			neuroUnitImp.setActivationFunction(acf);
//			neuros.add(neuroUnitImp); 
//			return;
//		}
		if (previousLayer == null) {
			updateValues4FirstLayer(input);
		} else {					//input[0].length	
			for (int i = 0; i < neuroCount; i++) {
//				NeuroUnitImp neuroUnitImp = new NeuroUnitImp();
				NeuroUnitImp neuroUnitImp = new NeuroUnitImpV2(this);
				neuroUnitImp.buildup(input, i);
				neuroUnitImp.setActivationFunction(acf);
				neuros.add(neuroUnitImp); 
			}			
		}
	}

	@Override
	public double getStdError(double [][] result) {
		double stdError = 0;
		for (int i = 0; i < result.length; i++) {
			double [] rzs = result[i];
			for (int j = 0; j < rzs.length; j++) {
				INeuroUnit neuro = getNeuros().get(j);
				double resisal = rzs[j] - neuro.getAaz(i);
				stdError = stdError + resisal * resisal;
			}
		}
//		INeuroUnit neuro = getNeuros().get(0);
//		double stdError = 0;
//		for (int i = 0; i < result.length; i++) {
//			double resisal = result[i] - neuro.getAaz(i);
//			stdError = stdError + resisal * resisal;
//		}
		return stdError/2.0;
	}

	@Override
	public void updateNeuros() {
		for (int i = 0; i < neuros.size(); i++) {
			INeuroUnit neuro = neuros.get(i);
			neuro.updateSelf();
		}
	}

}
