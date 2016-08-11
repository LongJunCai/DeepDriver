package deepDriver.dl.aml.cnn;

import deepDriver.dl.aml.ann.IActivationFunction;

public class SamplingReconstructionFeatureMap extends SamplingFeatureMap {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public SamplingReconstructionFeatureMap(ICNNLayer currentLayer, IActivationFunction acf,
			ICNNLayer previousLayer, int ckRows, int ckColumns,
			boolean isFullConnection, int[] previouFeatureMapSeq, int fmIndex) {
		super(currentLayer, acf, previousLayer, ckRows, ckColumns, isFullConnection,
				previouFeatureMapSeq, fmIndex);
	}
	
	public void resizeFeatures() {
		double [][] featureOfPrevious = previousLayer.getFeatureMaps()[0].getFeatures();
		//asume step = 1, and no need pending.
		int r = featureOfPrevious.length * ckRows;
		int c = featureOfPrevious[0].length * ckColumns;
		
		fmts = new FeatureMapTag[r][c];
		for (int i = 0; i < fmts.length; i++) {
			for (int j = 0; j < fmts[i].length; j++) {
				fmts[i][j] = new FeatureMapTag();
			}			
		}
		initFeatures(r, c);
	}

}
