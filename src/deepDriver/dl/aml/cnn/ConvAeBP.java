package deepDriver.dl.aml.cnn;


import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.cnn.cae.IConvAutoEncoderLayerVisitor;

public class ConvAeBP extends CNNBP implements IConvAutoEncoderLayerVisitor {

	public ConvAeBP(CNNConfigurator cfg) {
		super(cfg);
		this.fv = new ConvAeFV();
		fv.bp = this;
		wWUpv = new ConvAeWwUpdater();
		wWUpv.bp = this;
	}
	
	public void caculateStdError(ICNNLayer layer) {
		if (layerIndex == cfg.getLayers().length - 1) {
			double error = 0;
			double l = 0;
			IFeatureMap [] fmps = layer.getFeatureMaps();
			for (int i = 0; i < fmps.length; i++) {				
				double [][] f = fmps[i].getFeatures();
				double [][] zZ = fmps[i].getzZs();
				double [][] deltaZz = fmps[i].getDeltaZzs();
				l = fmps.length * f.length * f[0].length;
				double [][] m = this.dataMatrix[i].getMatrix();
				for (int j = 0; j < f.length; j++) {
					for (int j2 = 0; j2 < f[i].length; j2++) {
						double r = (f[j][j2] - m[j][j2]);
						deltaZz[j][j2] = r * fmps[i].getAcf().deActivate(zZ[j][j2]);
						error = error + r * r/2.0;
					}
				}
			}
			error = error/l;
			this.stdError = error;
//			System.out.println("The error is "+error);
		}
	}

	@Override
	public void visitCNNReconstructionLayer(CNNReconstructionLayer layer) {
		caculateStdError(layer);
		ICNNLayer previousLayer = layer.getPreviousLayer();
		if (previousLayer == null) {
			return; 
		}
		resetPreviousFlagMatrix(layer);
		IFeatureMap [] fms = layer.getFeatureMaps();		
		IFeatureMap[] fmsOfPrevious = previousLayer.getFeatureMaps();
		for (int i = 0; i < fms.length; i++) {			
			IConvolutionKernal [] cks = fms[i].getKernals();
			for (int j = 0; j < cks.length; j++) {
				ConvolutionKernal ck = (ConvolutionKernal)cks[j];
				ck.initB = false;
				resetFlagMatrix(ck.getInitDeltaZzs());
			}
		}
		for (int i = 0; i < fms.length; i++) {
			IConvolutionKernal[] cks = fms[i].getKernals();
			for (int j = 0; j < cks.length; j++) {
				IFeatureMap ffm = fmsOfPrevious[cks[j].getFmapOfPreviousLayer()];
				bpReconstructedConv(layer.getLc(), (ConvolutionKernal) cks[j], 
						fms[i], ffm, j == 0);
			}
			CNNUtils.deActivateGlobal(this, fms[i]);
		}
	
	}
	
	public void bpReconstructedConv(LayerConfigurator lc, ConvolutionKernal ck, IFeatureMap t2fm, IFeatureMap ffm2, boolean begin) {
		for (int i = 0; i < ffm2.getDeltaZzs().length; i++) {
			for (int j = 0; j < ffm2.getDeltaZzs()[i].length; j++) {	
				for (int k = 0; k < ck.wWs.length; k++) {
					for (int k2 = 0; k2 < ck.wWs[k].length; k2++) {
						int fr = i + k;
						int fc = j + k2;
						double dzZ = getConvFms(t2fm.getDeltaZzs(), fr, fc, lc);
						if (!ffm2.getInitDeltaZzs()[i][j]) {
							ffm2.getInitDeltaZzs()[i][j] = true;  
							ffm2.getDeltaZzs()[i][j] = dzZ * ck.wWs[k][k2];
						} else {
							double oz = ffm2.getDeltaZzs()[i][j];
							ffm2.getDeltaZzs()[i][j] = oz + dzZ * ck.wWs[k][k2];
						}
						//
						double aA = ffm2.getFeatures()[i][j];
						if (!ck.getInitDeltaZzs()[k][k2]) {
							ck.getInitDeltaZzs()[k][k2] = true;
							ck.detalwWs[k][k2] = cfg.getM() * ck.detalwWs[k][k2]
									- cfg.getL() * dzZ * aA;
						} else {
							ck.detalwWs[k][k2] = ck.detalwWs[k][k2] +
									- cfg.getL() * dzZ * aA;
						}						
					}
				}				
			}
		}
	}

	@Override
	public void visitPoolingReconstructionLayer(SamplingReconstructionLayer layer) {
		caculateStdError(layer);
		resetPreviousFlagMatrix(layer);
		IFeatureMap [] fms = layer.getFeatureMaps();
		IFeatureMap [] fmsInLastLayer = layer.getPreviousLayer().getFeatureMaps();
		for (int i = 0; i < fms.length; i++) {
			IConvolutionKernal [] cks = fms[i].getKernals();
			//in most cases, the ck is 1 - 1
			for (int j = 0; j < cks.length; j++) {
				SubSamplingKernal ssk = (SubSamplingKernal)cks[j];
				ssk.initwW = false;
				ssk.initB = false;
				IFeatureMap ffm = fmsInLastLayer[ssk.fmapOfPreviousLayer];
				bpReconstructedSampling(ssk, fms[i], ffm, fms[i].getAcf(), j == 0);
			}
			CNNUtils.deActivateGlobal(this, fms[i]);
		}
	
	}
	
	public void bpReconstructedSampling(SubSamplingKernal ck, IFeatureMap t2fm, IFeatureMap ffm2, IActivationFunction acf, 
			boolean begin) {
		for (int i = 0; i < ffm2.getDeltaZzs().length; i++) {
			for (int j = 0; j < ffm2.getDeltaZzs()[i].length; j++) {
				double cs = 0;
				for (int j2 = 0; j2 < ck.ckRows; j2++) {
					for (int k = 0; k < ck.ckColumns; k++) {
						int fr = i * ck.ckRows + j2;
						int fc = j * ck.ckColumns + k;
						/*auto padding
						 * **/
						if (fr >= t2fm.getFeatures().length || fc >= t2fm.getFeatures()[0].length) {
							continue;
						}/*auto padding
						 * **/
						double dzZ = ffm2.getDeltaZzs()[fr][fc];
						if (!ffm2.getInitDeltaZzs()[i][j]) {
							ffm2.getInitDeltaZzs()[i][j] = true;							
							ffm2.getDeltaZzs()[i][j] =  dzZ * ck.wW;
						} else {
							ffm2.getDeltaZzs()[i][j] = ffm2.getDeltaZzs()[i][j] + dzZ * ck.wW;
						}
						double aA = ffm2.getFeatures()[i][j];
						if (!ck.initwW) {
							ck.initwW = true;
							ck.deltawW = cfg.getM() * ck.deltawW 
									- cfg.getL() * aA * dzZ;
						} else {
							ck.deltawW = ck.deltawW 
									- cfg.getL() * aA * dzZ;
						}
					}
				}				
			}
		}
	}

}
