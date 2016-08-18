package deepDriver.dl.aml.ann;

import java.util.List;

public interface INeuroUnit {
	//theta[], result []	
	public double getAaz(int dataIndex);	
	
//	public int getInputSize();	
//	public double[] getDeltaZ();
	
	public double get4PropagationPreviousDelta(int dataIndex, int previouNeuroIndex);
	
	public void setActivationFunction(IActivationFunction activationFunction);
	
//	public void input(List<INeuroUnit> neuros);
	
	public void forwardPropagation(List<INeuroUnit> previousNeuros, double [][] inputs);
	
	public void backPropagation(List<INeuroUnit> previousNeuros, List<INeuroUnit> nextNeuros, double [][] finalResult, InputParameters parameters);
	
	public void buildup(double [][] input, int position);
	
	public void updateSelf();
}
