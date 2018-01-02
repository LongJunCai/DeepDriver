package deepDriver.dl.aml.common;

import deepDriver.dl.aml.costFunction.ICostFunction;

public interface ICommonLayer {
	
	public void setICostFunction(ICostFunction cf);
	
	public ICostFunction getICostFunction();
	
	
}
