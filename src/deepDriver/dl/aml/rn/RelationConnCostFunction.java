package deepDriver.dl.aml.rn;

import deepDriver.dl.aml.ann.ILayer;
import deepDriver.dl.aml.costFunction.ICostFunction;

public class RelationConnCostFunction implements ICostFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	RN4DNN rn;	

	public RelationConnCostFunction(RN4DNN rn) {
		super();
		this.rn = rn;
	}

	@Override
	public int getzZIndex() {
		return 0;
	}

	@Override
	public void setzZIndex(int zZIndex) {

	}

	@Override
	public double[] activate() {
		return null;
	}

	@Override
	public double caculateStdError() {
		return 0;
	}

	@Override
	public void caculateCostError() {

	}

	@Override
	public void setLayer(ILayer layer) {

	}

	@Override
	public void setTarget(double[] target) {

	}

	@Override
	public double verfiyResult(double[] targets, double[] results) {
		return 0;
	}

}
