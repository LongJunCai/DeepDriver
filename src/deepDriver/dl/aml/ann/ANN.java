package deepDriver.dl.aml.ann;

import java.io.Serializable;

import deepDriver.dl.aml.ann.imp.LayerImpV2;
import deepDriver.dl.aml.ann.imp.LogicsticsActivationFunction;
import deepDriver.dl.aml.costFunction.ICostFunction;

public class ANN implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	ILayer firstLayer;
	ILayer lastLayer;
	ICostFunction cf;	
	int kLength;
	
	String name;
	
	ANNCfg aNNCfg = new ANNCfg();	
	
	public ANNCfg getaNNCfg() {
		return aNNCfg;
	}

	public void setaNNCfg(ANNCfg aNNCfg) {
		this.aNNCfg = aNNCfg;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public ILayer getFirstLayer() {
		return firstLayer;
	}

	public void setFirstLayer(ILayer firstLayer) {
		this.firstLayer = firstLayer;
	}

	public ICostFunction getCf() {
		return cf;
	}

	public void setCf(ICostFunction cf) {
		this.cf = cf;
	}

	public ILayer createLayer() {
		LayerImpV2 layer = new LayerImpV2();
		layer.setaNNCfg(aNNCfg);
		if (cf != null) {
			layer.setCostFunction(cf);
		}
		return layer;
	}
	
	public IActivationFunction createActivation() {
		return new LogicsticsActivationFunction();
	}
	
	public void buildUp(int [] layerNodes) {
		firstLayer = createLayer();
		int firstLayerNodes = layerNodes[0];
		debugPrint("Begin to build up the ann:");
		IActivationFunction acf = createActivation();
		firstLayer.buildup(null, new double[][]{new double[firstLayerNodes]}, 
				acf, false, firstLayerNodes);
		ILayer tlayer = firstLayer;
		for (int i = 1; i < layerNodes.length; i++) {
			ILayer newLayer = createLayer();
			newLayer.setPos(i+1);
			newLayer.buildup(tlayer, null, acf, 
					i == layerNodes.length - 1, layerNodes[i]);
			tlayer = newLayer;
		}
		lastLayer = tlayer;
	}
	 
	public double[] forward(double [][] input) {
		ILayer layer = firstLayer;
		LayerImpV2 lastLayer = (LayerImpV2) this.lastLayer;
		while (layer != null) {
			debugPrint("ForwardPropagation on layer "+layer);
			layer.forwardPropagation(input);
			layer = layer.getNextLayer();
		}		
 		if (cf != null && lastLayer.getRs() != null) {
			return lastLayer.getRs();
		} else {
			return new double[] {lastLayer.getNeuros().get(0).getAaz(0)};
		}	 
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
	
	InputParameters inparams = new InputParameters();
	
	public double bp(double [][] result, double l, double m, double lamda) {
		inparams.setAlpha(l);
		inparams.setLamda(0);
		inparams.setM(m);
		double err = lastLayer.getStdError(result);
		ILayer layer = lastLayer;
		while (layer != null) {
			debugPrint("BackPropagation on layer "+layer);
			layer.backPropagation(result, inparams);
			lastLayer = layer;
			layer = layer.getPreviousLayer();
		}
		return err;
	}
	
	public void updateWws() {
		ILayer layer = firstLayer;
		while (layer != null) {
			debugPrint("update layer on layer "+layer);
			layer.updateNeuros();
			lastLayer = layer;
			layer = layer.getNextLayer();
		}
	}

	private void debugPrint(String string) {
//		System.out.println(string);		
	}
	
	

}
