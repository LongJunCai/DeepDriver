package deepDriver.dl.aml.costFunction;

import java.io.Serializable;

import deepDriver.dl.aml.ann.ILayer;

public interface ICostFunction extends Serializable {
	
	public double [] activate();
	
	public double caculateStdError();
	
	public void caculateCostError();
	
	public void setLayer(ILayer layer);

	public void setTarget(double[] target);
	
	public double verfiyResult(double [] targets, double [] results) ;

}
