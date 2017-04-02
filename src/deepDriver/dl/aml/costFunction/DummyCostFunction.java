package deepDriver.dl.aml.costFunction;

import java.io.Serializable;
import java.util.List;


import deepDriver.dl.aml.ann.ILayer;
import deepDriver.dl.aml.ann.INeuroUnit;
import deepDriver.dl.aml.ann.imp.LayerImp;
import deepDriver.dl.aml.ann.imp.NeuroUnitImp;

public class DummyCostFunction implements ICostFunction, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	LayerImp layer;
	
	int zZIndex = 0;
	
	public int getzZIndex() {
		return zZIndex;
	}

	public void setzZIndex(int zZIndex) {
		this.zZIndex = zZIndex;
	}

	@Override
	public double [] activate() {
		List<INeuroUnit> neuros = layer.getNeuros();
		double [] yt = new double[neuros.size()]; 
		for (int i = 0; i < neuros.size(); i++) { 
			NeuroUnitImp nu = (NeuroUnitImp) neuros.get(i);
			yt[i] = nu.getAas()[zZIndex]; 
		}		 
		return yt;
	}
	
	public double caculateStdError() { 
		return layer.getStdError(new double[][]{target});
	}

	double [] target;
	@Override
	public void caculateCostError() {
		
	}

	public LayerImp getLayer() {
		return layer;
	}

	public void setLayer(ILayer layer) {
		this.layer = (LayerImp) layer;
	}

	public double[] getTarget() {
		return target;
	}

	public void setTarget(double[] target) {
		this.target = target;
	}
	
	public static void main(String[] args) {
		System.out.println(Math.exp(-0.00928597904706061));
	}

	@Override
	public double verfiyResult(double[] targets, double[] results) {
		return 0;
	}
	
}