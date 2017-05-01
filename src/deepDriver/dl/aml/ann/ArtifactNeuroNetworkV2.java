package deepDriver.dl.aml.ann;

import java.io.Serializable;


import deepDriver.dl.aml.ann.imp.LayerImpV2;
import deepDriver.dl.aml.costFunction.ICostFunction;

public class ArtifactNeuroNetworkV2 extends ArtifactNeuroNetwork implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	ICostFunction cf;	
	int kLength;

	public int getkLength() {
		return kLength;
	}

	public void setkLength(int kLength) {
		this.kLength = kLength;
	}	

	public ICostFunction getCf() {
		return cf;
	}

	public void setCf(ICostFunction cf) {
		this.cf = cf;
	}
	
	public ILayer createLayerOnly() {
		LayerImpV2 layer = new LayerImpV2();
		layer.setaNNCfg(aNNCfg); 
		return layer;
	}

	@Override
	public ILayer createLayer() {
		LayerImpV2 layer = new LayerImpV2();
		layer.setaNNCfg(aNNCfg);
		if (cf != null) {
			layer.setCostFunction(cf);
		}
		return layer;
	}
	
	public double [][] getResults(InputParameters parameters) {
		double [][] result2 = parameters.getResult2();
		if (result2 != null) {
			return result2;
		}
		if (kLength <= 0) {
			kLength = 1;
		}
		double [] result = parameters.getResult();
		double [][] newResult = new double[result.length][kLength];
		if (kLength == 1) {//this is not a best method, but reduce the error issue possibility.
			for (int i = 0; i < newResult.length; i++) {
				newResult[i][0] = result[i];
			}
		} else {
			for (int i = 0; i < newResult.length; i++) {
				newResult[i] = new double[kLength];
				newResult[i][(int)result[i]] = 1;
			}
		}		
		return newResult;
	}
	
	public int getMaxPos(double [] x) {
		int mp = 0;
		for (int i = 0; i < x.length; i++) {
			if (x[mp] < x[i]) {
				mp = i;
			}
		}
		System.out.println(x[mp]);
		return mp;
	}
	
	public LayerImpV2 runResult(double [] x) {
		double [][] nx = new double[][] {x};
		LayerImpV2 lastLayer = (LayerImpV2) firstLayer;
		ILayer layer = firstLayer;
		firstLayer.buildup(null, nx, null, false, x.length);
		while (layer != null) {
			debugPrint("ForwardPropagation on layer "+layer);
			layer.forwardPropagation(nx);
			lastLayer = (LayerImpV2) layer;
			layer = layer.getNextLayer();
		}
		return lastLayer;
	}
	
	public double getResult(double [] x) {
		LayerImpV2 lastLayer = runResult(x);
		if (cf != null && lastLayer.getRs() != null) {
			return getMaxPos(lastLayer.getRs());
		} else {
			return lastLayer.getNeuros().get(0).getAaz(0);
		}		
	}
	
	ANNCfg aNNCfg = new ANNCfg();
	
	
	public ANNCfg getaNNCfg() {
		return aNNCfg;
	}

	public void setaNNCfg(ANNCfg aNNCfg) {
		this.aNNCfg = aNNCfg;
	}
	
	public double [][] testModel2(InputParameters parameters) {
		aNNCfg.isTesting = true;
		debugPrint("Begin to build up the ann for test:");
		double [][] input = normalizer.retransformParameters(parameters.getInput());		
		double arr [][] = new double[input.length][];
		for (int i = 0; i < arr.length; i++) {
			LayerImpV2 lastLayer = runResult(input[i]);
			arr[i] = lastLayer.getRs();
		}
		aNNCfg.isTesting = false;
		return arr;
	}

	public double [] testModel(InputParameters parameters) {
		aNNCfg.isTesting = true;
		debugPrint("Begin to build up the ann for test:");
		double [][] input = normalizer.retransformParameters(parameters.getInput());		
		double arr [] = new double[input.length];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = getResult(input[i]);
		}
		aNNCfg.isTesting = false;
		return arr;
	}
	
}
