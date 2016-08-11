package deepDriver.dl.aml.lstm.imp;

import java.io.Serializable;

import deepDriver.dl.aml.lstm.IBlock;
import deepDriver.dl.aml.lstm.ICell;
import deepDriver.dl.aml.lstm.IForgetGate;
import deepDriver.dl.aml.lstm.IInputGate;
import deepDriver.dl.aml.lstm.IOutputGate;
import deepDriver.dl.aml.lstm.RNNNeuroVo;

public class Block implements IBlock, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public Block(int layerNN, int blockNN, int t, boolean inHidenLayer, int previousNNN, int nextLayerNN) {
		this.inputGate = new InputGate(t, inHidenLayer, previousNNN, layerNN, blockNN, nextLayerNN);
		this.outPutGate = new OutputGate(t, inHidenLayer, previousNNN, layerNN, blockNN, nextLayerNN);
		this.forgetGate = new ForgetGate(t, inHidenLayer, previousNNN, layerNN, blockNN, nextLayerNN);
		
		cells = new Cell[blockNN];
		for (int i = 0; i < cells.length; i++) {
			cells[i] = new Cell(t, inHidenLayer, previousNNN, layerNN, blockNN, nextLayerNN);
			//it is said state should be 1 initialization
			double [] sc = new double[t];
			for (int j = 0; j < sc.length; j++) {
				sc[j] = 1;
			}
			cells[i].setSc(sc);
			cells[i].setDeltaSc(new double[t]);
			
			cells[i].setCZz(new double[t]);
			cells[i].setDeltaC(new double[t]);
		}
	}
	
	InputGate inputGate;
	OutputGate outPutGate;
	ForgetGate forgetGate;
	Cell [] cells;
	public IInputGate getInputGate() {
		return inputGate;
	}
	public void setInputGate(IInputGate inputGate) {
		this.inputGate = (InputGate) inputGate;
	}
	public IOutputGate getOutPutGate() {
		return outPutGate;
	}
	public void setOutPutGate(IOutputGate outPutGate) {
		this.outPutGate = (OutputGate) outPutGate;
	}
	public IForgetGate getForgetGate() {
		return forgetGate;
	}
	public void setForgetGate(IForgetGate forgetGate) {
		this.forgetGate = (ForgetGate) forgetGate;
	}
	public ICell[] getCells() {
		return cells;
	}
	public void setCells(ICell[] cells) {
		this.cells = (Cell[]) cells;
	}
	
	public RNNNeuroVo [] getRNNNeuroVos() {
		return cells;
	} 

}
