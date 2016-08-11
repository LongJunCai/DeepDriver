package deepDriver.dl.aml.ann.imp;

import java.util.List;

import deepDriver.dl.aml.ann.INeuroUnit;
import deepDriver.dl.aml.ann.ISparseAutoEncoderCfg;
import deepDriver.dl.aml.ann.InputParameters;

public class SparseAutoEncoderNeuro extends NeuroUnitImpV3 {

	public SparseAutoEncoderNeuro(LayerImp layer) {
		super(layer);
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public SparseAutoEncoderLayer getSparseAutoEncoderLayer() {		
		return (SparseAutoEncoderLayer) layer;
	}
	
	public ISparseAutoEncoderCfg getSparseAutoEncoderCfg() {		
		return getSparseAutoEncoderLayer().getSparseAutoEncoderCfg();
	}
	
	@Override
	public void backPropagation(List<INeuroUnit> previousNeuros, List<INeuroUnit> nextNeuros, double [][] result, InputParameters parameters) {
		super.backPropagation(previousNeuros, nextNeuros, result, parameters);
		ISparseAutoEncoderCfg cfg = getSparseAutoEncoderCfg();
		if (cfg == null) {
			return ;
		}
		if (nextNeuros == null) {
		} else {
			if (layer.getPreviousLayer() == null) {
				return ;
			}			
			for (int i = 0; i < deltaZ.length; i++) {
				double sumDelta = 0;
				double p = cfg.getP();
				double p1 = aas[i];
				sumDelta = - p/p1 + (1- p)/(1 - p1);
				deltaZ[i] = deltaZ[i] +cfg.getBeta() * (sumDelta) * activationFunction.deActivate(zzs[i]);
			}
		}
		
	}

}
