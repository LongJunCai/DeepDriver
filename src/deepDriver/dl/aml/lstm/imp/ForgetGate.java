package deepDriver.dl.aml.lstm.imp;

import java.io.Serializable;

import deepDriver.dl.aml.lstm.IForgetGate;
import deepDriver.dl.aml.lstm.RNNNeuroVo;

public class ForgetGate extends RNNNeuroVo implements IForgetGate, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public ForgetGate(int t, boolean inHidenLayer, int previousNNN,
			int LayerNN, int blockNN, int nextLayerNN) {
		super(t, inHidenLayer, previousNNN, LayerNN, blockNN, nextLayerNN);
	}

}
