package deepDriver.dl.aml.ann;

import java.util.List;

public interface ILayer {
	
	public int getPos();

	public void setPos(int pos);
	
	public void setNextLayer(ILayer iLayer);
	
	public void setPreviousLayer(ILayer iLayer);
	
	public ILayer getNextLayer();
	
	public ILayer getPreviousLayer();
	
	public void addNeuro(INeuroUnit neuro);
	
	public List<INeuroUnit> getNeuros();
	
	public void buildup(ILayer previousLayer, double [][] input, IActivationFunction acf
			, boolean isLastLayer, int neuroCount);
	
	public void forwardPropagation(double [][] input);
	
	public void backPropagation(double [][] finalResult, InputParameters parameters);

	public void updateNeuros();
	
	public double getStdError(double [][] result);
}
