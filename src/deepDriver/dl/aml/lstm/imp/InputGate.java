package deepDriver.dl.aml.lstm.imp;

import java.io.Serializable;

import deepDriver.dl.aml.lstm.IInputGate;
import deepDriver.dl.aml.lstm.RNNNeuroVo;

public class InputGate extends RNNNeuroVo implements IInputGate, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public InputGate(int t, boolean inHidenLayer, int previousNNN,
			int LayerNN, int blockNN, int nextLayerNN) {
		super(t, inHidenLayer, previousNNN, LayerNN, blockNN, nextLayerNN, null); 
	}

}
