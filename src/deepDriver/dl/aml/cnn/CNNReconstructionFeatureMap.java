package deepDriver.dl.aml.cnn;

import deepDriver.dl.aml.ann.IActivationFunction;

public class CNNReconstructionFeatureMap extends FeatureMap {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;	

	public CNNReconstructionFeatureMap(ICNNLayer currentLayer, IActivationFunction acf,
			ICNNLayer previousLayer, int ckRows, int ckColumns,
			boolean isFullConnection, int[] previouFeatureMapSeq, int fmIndex) {
		super(currentLayer, acf, previousLayer, ckRows, ckColumns, isFullConnection,
				previouFeatureMapSeq, fmIndex);
	}
	
	public void resizeFeatures() {
		// pfm = cfm - 1 - padding + ck
		double [][] featureOfPrevious = previousLayer.getFeatureMaps()[0].getFeatures();
		int padding = 2 * previousLayer.getLc().getPadding();
		//asume step = 1, and no need padding.
		int r = - padding + featureOfPrevious.length + ckRows - 1;
		int c = - padding + featureOfPrevious[0].length + ckColumns - 1;
		initFeatures(r, c);	
	}

}
