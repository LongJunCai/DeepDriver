package deepDriver.dl.aml.lstm;

import java.io.Serializable;

public class NeuroNetworkArchitecture implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	int [] nnArch;
	int costFunction = LSTMConfigurator.LEAST_SQUARE;
		
	boolean useProjectionLayer = false;	
	
	public static int HiddenLSTM = 1;
	public static int HiddenRNN = 2;
	
	int hiddenType = HiddenLSTM;
	
	

	public int getHiddenType() {
		return hiddenType;
	}

	public void setHiddenType(int hiddenType) {
		this.hiddenType = hiddenType;
	}

	public boolean isUseProjectionLayer() {
		return useProjectionLayer;
	}

	public void setUseProjectionLayer(boolean useProjectionLayer) {
		this.useProjectionLayer = useProjectionLayer;
	}

	public int[] getNnArch() {
		return nnArch;
	}

	public void setNnArch(int[] nnArch) {
		this.nnArch = nnArch;
	}

	public int getCostFunction() {
		return costFunction;
	}

	public void setCostFunction(int costFunction) {
		this.costFunction = costFunction;
	}
	

}
