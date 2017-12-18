package deepDriver.dl.aml.ann.imp;

import java.io.Serializable;
import java.util.List;

import deepDriver.dl.aml.ann.INeuroUnit;
import deepDriver.dl.aml.ann.ISparseAutoEncoderCfg;

public class SparseAutoEncoderLayer extends LayerImpV2 implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public ISparseAutoEncoderCfg getSparseAutoEncoderCfg() {
		if (getaNNCfg() instanceof ISparseAutoEncoderCfg) {
			return (ISparseAutoEncoderCfg) getaNNCfg();			
		}
		return null;
	}
	
	public NeuroUnitImp createNeuroUnitImp() {
		return new SparseAutoEncoderNeuro(this);
	}
	
	int zZIndex = 0;
	@Override
	public double getStdError(double[][] result) {
		//since this is a sparse auto encoder, so assume it is a 3-layer one
		ISparseAutoEncoderCfg cfg = getSparseAutoEncoderCfg();
		if (cfg == null || getPreviousLayer().getPreviousLayer() == null) {
			return super.getStdError(result);
		}
		List<INeuroUnit> list = getPreviousLayer().getNeuros();		
		double kl = 0;
		double p = cfg.getP();
		for (int i = 0; i < list.size(); i++) {
			INeuroUnit nu = list.get(i);
			
//			double [] aAs = nu.getAas();
//			for (int j = 0; j < aAs.length; j++) {
//				
//			}
			double p1 = nu.getAaz(zZIndex);
			kl = kl + p * Math.log(p/p1) + (1 - p) * Math.log((1-p)/(1- p1));
		}
		return super.getStdError(result) + cfg.getBeta() * kl;
	}

}
