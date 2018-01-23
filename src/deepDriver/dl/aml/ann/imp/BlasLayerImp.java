package deepDriver.dl.aml.ann.imp;

import java.util.List;

import deepDriver.dl.aml.ann.ILayer;
import deepDriver.dl.aml.ann.INeuroUnit;
import deepDriver.dl.aml.ann.InputParameters;
import deepDriver.dl.aml.math.MathUtil;

public class BlasLayerImp {
	ILayer layer;
	double [] aAs = null;
	double [][] wWs = null;
	double [][] dwWs = null;
	public void forwardPropagation(double [][] input) {
		if (wWs == null) {
			List<INeuroUnit> nus = layer.getNeuros();
			aAs = new double[nus.size()];
			wWs = new double[nus.size()][];
			dwWs = new double[wWs.length][];
			for (int i = 0; i < nus.size(); i++) {
				INeuroUnit nu = nus.get(i);
				wWs[i] = nu.getThetas();
				dwWs[i] = new double[wWs[i].length];
			}
		}
		ILayer pl = layer.getPreviousLayer();
		List<INeuroUnit> nus = pl.getNeuros();
		double [] pvo = new double[nus.size() + 1];
		for (int i = 0; i < nus.size(); i++) {
			INeuroUnit nu = nus.get(i);
			pvo[i] = nu.getAaz(0);
		}
		pvo[pvo.length - 1] = 1.0;
		aAs = MathUtil.transpose(MathUtil.multiple(wWs, MathUtil.transpose(new double[][]{pvo})))[0];
		
		nus = layer.getNeuros(); 
		for (int i = 0; i < nus.size(); i++) {
			INeuroUnit nu = nus.get(i);
//			nu.get
		}
		
	}
	
	public void backPropagation(double [][] finalResult, InputParameters parameters) {
		
	}

}
