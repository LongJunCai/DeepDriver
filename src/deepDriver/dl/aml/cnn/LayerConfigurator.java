package deepDriver.dl.aml.cnn;

import java.io.Serializable;


import deepDriver.dl.aml.ann.ANNCfg;
import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.costFunction.ICostFunction;

public class LayerConfigurator implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public static int CONVOLUTION_LAYER = 1;
	public static int POOLING_LAYER = 2;
	public static int ANN_LAYER = 3;
	public static int FRACTAL_BLOCK_LAYER = 4;
	
	public static int CONV_RECONSTRUCTION_LAYER = 4;
	public static int SAMPLING_RECONSTRUCTION_LAYER = 5;
	
	public static int FM_ALLOCATED = 1;
	public static int FM_NON_ALLOCATED = 0;
	
	int fblockDepth;
	int fblockLayerNum;
	
	int type = CONVOLUTION_LAYER;
	
	boolean resNetLayer;
	
	int padding = 0;
	
	int featureMapNum;
	boolean isFullConnection;
	int [][] featureMapAllocationMatrix;
	int [][] cks;
	
	boolean isLast;
	
	int ckRows;
	int ckColumns;
	
	int step = 1;
		
	ANNCfg aNNCfg;
	
	IActivationFunction acf;
	
	boolean isFMAdaptive;
	boolean isCKAdaptive;

	public LayerConfigurator(int type, int featureMapNum,
			boolean isFullConnection, int ckWidth, int ckHeight, int step) {
		super();
		this.type = type;
		this.featureMapNum = featureMapNum;
		this.isFullConnection = isFullConnection;
		this.ckRows = ckWidth;
		this.ckColumns = ckHeight;
		this.step = step;
	}				
	
	public boolean isResNetLayer() {
		return resNetLayer;
	}
	
	public void setResNetLayer(boolean resNetLayer) {
		this.resNetLayer = resNetLayer;
	}

	public int getFblockDepth() {
		return fblockDepth;
	}

	public void setFblockDepth(int fblockDepth) {
		this.fblockDepth = fblockDepth;
	}

	public int getFblockLayerNum() {
		return fblockLayerNum;
	}

	public void setFblockLayerNum(int fblockLayerNum) {
		this.fblockLayerNum = fblockLayerNum;
	}

	public int getPadding() {
		return padding;
	}
	
	public void setPadding(int padding) {
		this.padding = padding;
	}

	public boolean isFMAdaptive() {
		return isFMAdaptive;
	}
	public void setFMAdaptive(boolean isFMAdaptive) {
		this.isFMAdaptive = isFMAdaptive;
	}



	public boolean isCKAdaptive() {
		return isCKAdaptive;
	}



	public void setCKAdaptive(boolean isCKAdaptive) {
		this.isCKAdaptive = isCKAdaptive;
	}



	public int[][] getCks() {
		return cks;
	}

	public void setCks(int[][] cks) {
		this.cks = cks;
	}

	public ANNCfg getaNNCfg() {
		return aNNCfg;
	}

	public void setaNNCfg(ANNCfg aNNCfg) {
		this.aNNCfg = aNNCfg;
	}

	ICostFunction costFunction;
	
	public ICostFunction getCostFunction() {
		return costFunction;
	}

	public void setCostFunction(ICostFunction costFunction) {
		this.costFunction = costFunction;
	}

	public IActivationFunction getAcf() {
		return acf;
	}
	
	public void setAcf(IActivationFunction acf) {
		this.acf = acf;
	}

	public int getType() {
		return type;
	}

	public void setType(int type) {
		this.type = type;
	}

	public int getFeatureMapNum() {
		return featureMapNum;
	}

	public void setFeatureMapNum(int featureMapNum) {
		this.featureMapNum = featureMapNum;
	}

	public boolean isFullConnection() {
		return isFullConnection;
	}

	public void setFullConnection(boolean isFullConnection) {
		this.isFullConnection = isFullConnection;
	}

	public int[][] getFeatureMapAllocationMatrix() {
		return featureMapAllocationMatrix;
	}

	public void setFeatureMapAllocationMatrix(int[][] featureMapAllocationMatrix) {
		this.featureMapAllocationMatrix = featureMapAllocationMatrix;
	}

	

	public int getCkRows() {
		return ckRows;
	}

	public void setCkRows(int ckRows) {
		this.ckRows = ckRows;
	}

	public int getCkColumns() {
		return ckColumns;
	}

	public void setCkColumns(int ckColumns) {
		this.ckColumns = ckColumns;
	}

	public int getStep() {
		return step;
	}

	public void setStep(int step) {
		this.step = step;
	}

	public boolean isLast() {
		return isLast;
	}

	public void setLast(boolean isLast) {
		this.isLast = isLast;
	}	

}
