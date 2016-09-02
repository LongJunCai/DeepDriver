package deepDriver.dl.aml.lstm.imp;

import java.io.Serializable;

import deepDriver.dl.aml.lstm.IOutputGate;
import deepDriver.dl.aml.lstm.RNNNeuroVo;

public class OutputGate extends RNNNeuroVo implements IOutputGate, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public OutputGate(int t, boolean inHidenLayer, int previousNNN,
			int LayerNN, int blockNN, int nextLayerNN) {
		super(t, inHidenLayer, previousNNN, LayerNN, blockNN, nextLayerNN, null);
	}

}
