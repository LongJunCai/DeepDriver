package deepDriver.dl.aml.lstm;

public interface IBlock {
	
	public IInputGate getInputGate();
	
	public void setInputGate(IInputGate inputGate);
	
	public IOutputGate getOutPutGate();
	
	public void setOutPutGate(IOutputGate outPutGate);
	
	public IForgetGate getForgetGate();
	
	public void setForgetGate(IForgetGate forgetGate);
	
	public ICell[] getCells();
	
	public void setCells(ICell[] cells);

}
