package deepDriver.dl.aml.cnn;


import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.cnn.cae.IConvAutoEncoderLayerVisitor;

public class ConvAeFV extends CNNForwardVisitor implements IConvAutoEncoderLayerVisitor {

	@Override
	public void visitCNNReconstructionLayer(CNNReconstructionLayer layer) {	
		IFeatureMap [] fms = layer.getFeatureMaps();
		IFeatureMap [] fmsInLastLayer = layer.getPreviousLayer().getFeatureMaps();			
		for (int i = 0; i < fms.length; i++) {
			//<adaptive the feature map>
			if (layer.getLc().isFMAdaptive()) {
				fms[i].resizeFeatures();
			}
			//<adaptive the feature map>
			fms[i].reset();
			IConvolutionKernal[] cks = fms[i].getKernals();
			for (int j = 0; j < cks.length; j++) {
				IFeatureMap ffm = fmsInLastLayer[cks[j].getFmapOfPreviousLayer()];
				reconstructConv(layer.getLc(), (ConvolutionKernal)cks[j], 
						(FeatureMap)fms[i], (FeatureMap)ffm, j == 0);
			}
			activateZzs(fms[i]);
		}
	}
	
	public void reconstructConv(LayerConfigurator lc, ConvolutionKernal ck, FeatureMap t2fm, FeatureMap ffms2, boolean begin) {
		for (int i = 0; i < ffms2.getFeatures().length; i++) {
			for (int j = 0; j < ffms2.getFeatures()[i].length; j++) {				
				double cs = 0;
				for (int k = 0; k < ck.wWs.length; k++) {
					for (int k2 = 0; k2 < ck.wWs[k].length; k2++) {
//						cs = cs + ffms[i + k][j + k2] * ck.wWs[k][k2];
//						cs = cs + bp.getConvFms(t2fm, i + k, j + k2, lc) * ck.wWs[k][k2];
						cs = bp.getConvFms(t2fm.getzZs(), i + k, j + k2, lc);
						bp.setConvFms(lc, t2fm.getzZs(), i + k, j + k2, 
								cs+ ffms2.getzZs()[i][j] * ck.wWs[k][k2]);
					}
				}
//				if (!bp.useGlobalWeight) {
//					cs = cs + ck.b;
//				}
			}
		}
	}

	@Override
	public void visitPoolingReconstructionLayer(
			SamplingReconstructionLayer layer) {
		IFeatureMap [] fms = layer.getFeatureMaps();
		IFeatureMap [] fmsInLastLayer = layer.getPreviousLayer().getFeatureMaps();
		for (int i = 0; i < fms.length; i++) {
			fms[i].reset();
			IConvolutionKernal[] cks = fms[i].getKernals();
			//in most cases, the ck is 1 - 1
			for (int j = 0; j < cks.length; j++) {
				SubSamplingKernal ssk = (SubSamplingKernal)cks[j];
				IFeatureMap ffm = fmsInLastLayer[cks[j].getFmapOfPreviousLayer()];
				//<adapt to ck>
				if (layer.getLc().isCKAdaptive()) {
					ssk.ckRows = ffm.getFeatures().length - fms[i].getFeatures().length + 1;
					ssk.ckColumns = ffm.getFeatures()[0].length - fms[i].getFeatures()[0].length + 1;
				}
				//<adapt to ck>				
				reconstructPooling(ssk, fms[i].getzZs(), ffm, j == 0, fms[i].getAcf());
			}
			activateZzs(fms[i]);
		}		
	}
	
	public void reconstructPooling(SubSamplingKernal ck, double [][] t2fm, IFeatureMap ffms2, boolean begin, IActivationFunction acf) {
		for (int i = 0; i < ffms2.getFeatures().length; i++) {
			for (int j = 0; j < ffms2.getFeatures()[i].length; j++) {
				double cs = 0;
				for (int j2 = 0; j2 < ck.ckRows; j2++) {
					for (int k = 0; k < ck.ckColumns; k++) {
						int fr = i * ck.ckRows + j2;
						int fc = j * ck.ckColumns + k;
						/*auto padding
						 * **/
						if (fr >= t2fm.length || fc >= t2fm[0].length) {
							continue;
						}/*auto padding
						 * **/
						t2fm[fr][fc] =  t2fm[fr][fc] + ffms2.getFeatures()[i][j] * ck.wW;
					}
				}
				if (!bp.useGlobalWeight) {
					cs = cs + ck.b;
				}				
			}
		}
	}

}
