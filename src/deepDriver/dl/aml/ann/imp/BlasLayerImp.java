package deepDriver.dl.aml.ann.imp;

import java.util.List;

import deepDriver.dl.aml.ann.ILayer;
import deepDriver.dl.aml.ann.INeuroUnit;
import deepDriver.dl.aml.ann.InputParameters;
import deepDriver.dl.aml.math.MathUtil;

public class BlasLayerImp {
	ILayer layer;
	double [] aAs = null;
	double [] zZs = null;
	double [][] wWs = null;
	double [][] dwWs = null;
	public void forwardPropagation(double [][] input) {
		if (wWs == null) {
			List<INeuroUnit> nus = layer.getNeuros();
			aAs = new double[nus.size()];
			zZs = new double[nus.size()];
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
			pvo[i] = nu.getAaz(zi);
		}
		pvo[pvo.length - 1] = 1.0;
		zZs = MathUtil.transpose(MathUtil.multiple(wWs, MathUtil.transpose(new double[][]{pvo})))[0];
		
		nus = layer.getNeuros(); 
		for (int i = 0; i < nus.size(); i++) {
			NeuroUnitImpV3 nu = (NeuroUnitImpV3) nus.get(i);
			aAs[i] = nu.getActivationFunction().activate(zZs[i]);
			nu.getAas()[0] = aAs[i];
		}
		
	}
	
	int zi = 0;
	
	public void backPropagation(double [][] finalResult, InputParameters parameters) {
//		ILayer nl = layer.getNextLayer(); 
		
		List<INeuroUnit> nus = layer.getNeuros();
		double [] dzZs = new double[nus.size() + 1];
		if (layer.getNextLayer() != null) {
			for (int i = 0; i < nus.size(); i++) {
				NeuroUnitImpV3 nu = (NeuroUnitImpV3) nus.get(i);
				dzZs[i] = nu.getDeltaZ()[zi] * nu.getActivationFunction().deActivate(zZs[i]);
			}
		}		
		ILayer pl = layer.getPreviousLayer();
		List<INeuroUnit> pnus = pl.getNeuros();
//		double [] pdZzs = new double[pnus.size()];
		double [] povs = new double[pnus.size()];
		for (int i = 0; i < pnus.size(); i++) {
			NeuroUnitImpV3 nu = (NeuroUnitImpV3) pnus.get(i);
//			pdZzs[i] = nu.getDeltaZ()[zi];
			povs[i] = nu.getAaz(zi);
		}
		
		//wWs * pAas = Aas
		dwWs = MathUtil.difMultipleX(dzZs, povs);
		
		double [] pdZzs = MathUtil.difMultipleY2v(dzZs, wWs);
		for (int i = 0; i < pnus.size(); i++) {
			NeuroUnitImpV3 nu = (NeuroUnitImpV3) pnus.get(i);
			nu.getDeltaZ()[zi] = pdZzs[i]; 
		}
	}
	
	public void updateNeuros() {
//		double [] povs = new double[pnus.size()];
//		for (int i = 0; i < pnus.size(); i++) {
//			NeuroUnitImpV3 nu = (NeuroUnitImpV3) pnus.get(i);
////			pdZzs[i] = nu.getDeltaZ()[zi];
//			povs[i] = nu.getAaz(zi);
//		}
//		//wWs * pAas = Aas
//		dwWs = MathUtil.difMultipleX(dzZs, povs);
	}

}
