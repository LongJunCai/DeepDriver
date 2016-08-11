package deepDriver.dl.aml.cnn;

import java.io.Serializable;

import deepDriver.dl.aml.ann.IActivationFunction;

public class SamplingFeatureMap extends FeatureMap implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public SamplingFeatureMap(ICNNLayer clayer, IActivationFunction acf, ICNNLayer previousLayer, int ckRows,
			int ckColumns, boolean isFullConnection, int[] previouFeatureMapSeq, int fmIndex) {		
		super(clayer, acf, previousLayer, ckRows, ckColumns, isFullConnection, previouFeatureMapSeq, fmIndex);
	}
	
	FeatureMapTag [][] fmts;
	
	public FeatureMapTag[][] getFmts() {
		return fmts;
	}
	public void setFmts(FeatureMapTag[][] fmts) {
		this.fmts = fmts;
	}
	
	public void initWwBase() {
//		double [][] featureOfPrevious = previousLayer.getFeatureMaps()[0].getFeatures();
////		double b = Math.pow(6.0/(double)(
////				featureOfPrevious.length * featureOfPrevious[0].length
////				+  features.length * features[0].length), 0.5);
//		double b = Math.pow(4.0/(double)(ckRows * ckColumns + 1), 0.5);
//		length = 2*b;
//		min = -b;
//		max = b;
	}
	
	public void resizeFeatures() {
		double [][] featureOfPrevious = previousLayer.getFeatureMaps()[0].getFeatures();
		//asume step = 1, and no need pending.
		int r = featureOfPrevious.length/ckRows;
		if (r * ckRows < featureOfPrevious.length) {
			r = r + 1;
		}
		int c = featureOfPrevious[0].length/ckColumns;
		if (c * ckColumns < featureOfPrevious[0].length) {
			c = c + 1;
		}
		/***use adaptive ck instead
		 * **/
		if (currentLayer.getLc().isCKAdaptive()) {
            r = ckRows;
            c = ckColumns;
        }
		/***use adaptive ck instead
         * **/
		fmts = new FeatureMapTag[r][c];
		for (int i = 0; i < fmts.length; i++) {
			for (int j = 0; j < fmts[i].length; j++) {
				fmts[i][j] = new FeatureMapTag();
			}			
		}
		initFeatures(r, c);
//		features = new double[r][c];
//		deltaZzs = new double[r][c];		
//		initDeltaZzs = new boolean[r][c];
//		for (int i = 0; i < features.length; i++) {
//			features[i] = new double[c];
//			deltaZzs[i] = new double[c];
//			initDeltaZzs[i] = new boolean[c];
//		}	
	}
	
	
	
	public void initCks() {
		if (isFullConnection) {
			kernals = new IConvolutionKernal[1];
			for (int i = 0; i < kernals.length; i++) {
				kernals[i] = createIConvolutionKernal();
				kernals[i].setFmapOfPreviousLayer(fmIndex);
			}
		} else {
			super.initCks();
		}		
	}
	
	public IConvolutionKernal createIConvolutionKernal() {
		SubSamplingKernal ck = new SubSamplingKernal();
		ck.wW = length * random.nextDouble()
				+ min;
		ck.b = length * random.nextDouble()
				+ min;
		ck.ckRows = ckRows;
		ck.ckColumns = ckColumns;
		return ck;
	}

}
