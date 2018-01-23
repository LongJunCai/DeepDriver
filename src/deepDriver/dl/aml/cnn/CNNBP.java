package deepDriver.dl.aml.cnn;

import java.util.List;


import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.ann.INeuroUnit;
import deepDriver.dl.aml.ann.InputParameters;
import deepDriver.dl.aml.ann.imp.NeuroUnitImp;
import deepDriver.dl.aml.costFunction.SoftMax4ANN;
import deepDriver.dl.aml.distribution.modelParallel.PartialCallback;
import deepDriver.dl.aml.distribution.modelParallel.ThreadParallel;

public class CNNBP implements ICNNBP {
	
	protected CNNConfigurator cfg;
	protected CNNForwardVisitor fv = new CNNForwardVisitor();
	protected CNNWwUpdateVisitor wWUpv = new CNNWwUpdateVisitor();
	protected SoftMax4ANN softMax = new SoftMax4ANN();
	
	public CNNBP(CNNConfigurator cfg) {
		super();
		this.cfg = cfg;
		fv.bp = this;
		wWUpv.bp = this;
	}
	
	public boolean useBlas() {
		return true;
	}
	
	
	public double getStdError() {
		return stdError;
	}

	public void setStdError(double stdError) {
		this.stdError = stdError;
	}

	boolean useGlobalWeight = true;

	double stdError = 0;

	public double runTrainEpich(IDataMatrix [] dataMatrix, double [] target) {
		this.target = target;
		fwd4(dataMatrix);
		bp();
		updateWws();
		return stdError;
	}
	
	double [] result;	
	
	public double[] getResult() {
		return result;
	}

	public void setResult(double[] result) {
		this.result = result;
	}

	public double[] test(IDataMatrix [] dataMatrix) {
		fwd4(dataMatrix);
		return result;
	}
	
	int layerIndex = 0;
	/**
	 * sometimes there are several cmfs in the 1st layer.
	 * and its target is softmax.
	 * **/
	IDataMatrix [] dataMatrix;	
	double [] target;
	public void fwd4(IDataMatrix [] dataMatrix) {	
		this.dataMatrix = dataMatrix;
		for (int i = 0; i < cfg.getLayers().length; i++) {
			layerIndex = i;
			cfg.getLayers()[i].accept(fv);
		}
	}	
	
	public void bp() {
		for (int i = cfg.getLayers().length - 1; i >= 0 ; i--) {
			layerIndex = i;
			cfg.getLayers()[i].accept(this);
		}
	}
	
	public void updateWws() {
		for (int i = 0; i < cfg.getLayers().length; i++) {
			layerIndex = i;
			cfg.getLayers()[i].accept(wWUpv);
		}
	}
	
	public boolean useVisitFractalBlock(CNNLayer layer) {
		if (layer instanceof FractalBlock) {
			FractalBlock fb = (FractalBlock) layer;
			if (fb.getDirectLayer() != null || fb.isResNet()) {
				return true;
			}			
		}
		return false;
	}
	
//	public boolean hasDirectLayerOrResLayer(FractalBlock fb) {
//		if (fb.getDirectLayer() != null || fb.isResNet()) {
//			return true;
//		}
//		return false;
//	}
	
//	boolean useBN = false;

	BlasCNNBpVisitor blasbp;
	@Override
	public void visitCNNLayer(CNNLayer layer) {
		/***visit block
		 * **/
		if (useVisitFractalBlock(layer)) {
			FractalBlock block = (FractalBlock) layer;
			visitFractualBlock(block);
			return;
		}
		/***visit block
		 * **/
		ICNNLayer previousLayer = layer.getPreviousLayer();
		if (previousLayer == null) {
			return; 
		}
		resetPreviousFlagMatrix(layer);
		IFeatureMap [] fms = layer.getFeatureMaps();		
		IFeatureMap[] fmsOfPrevious = previousLayer.getFeatureMaps();
		
		/***Use blas to speed up***/
		if (useBlas()) {
			if (this.blasbp == null) {
				blasbp = new BlasCNNBpVisitor(this);
			}
			blasbp.visitCNNLayer(layer);
			return;
		}
		/***Use blas to speed up***/
//		visitPartialCNNLayer(fms, fmsOfPrevious, layer, previousLayer);
		visitPartialCNNLayer2(fms, fmsOfPrevious, layer, previousLayer);
	}
	
	ThreadParallel threadParallel = new ThreadParallel();
	boolean useParallel = true;
	
	public void visitPartialCNNLayer2(final IFeatureMap [] fms, final IFeatureMap[] fmsOfPrevious, final CNNLayer layer, final ICNNLayer previousLayer) {
		int tn = cfg.getThreadsNum();
		if (tn <= 1 || !useParallel) {
			deActivateDZz4PartialCurrentLayer(fms, fmsOfPrevious, layer, previousLayer,
				0, fms.length);
		} else {
			threadParallel.runMutipleThreads(fms.length, new PartialCallback() {
				public void runPartial(int offset, int runLen) {
					deActivateDZz4PartialCurrentLayer(fms, fmsOfPrevious, layer, previousLayer,
							offset, runLen);
				}
			}, tn);
		}
		
		if (tn <= 1 || !useParallel) {
			bp4PartialPreviousLayer(fms, fmsOfPrevious, layer, previousLayer,
				0, fmsOfPrevious.length);
		} else {
			threadParallel.runMutipleThreads(fmsOfPrevious.length, new PartialCallback() {
				public void runPartial(int offset, int runLen) {
					bp4PartialPreviousLayer(fms, fmsOfPrevious, layer, previousLayer,
							offset, runLen);
				}
			}, tn);
		}
	}
	public void deActivateDZz4PartialCurrentLayer(IFeatureMap [] fms, IFeatureMap[] fmsOfPrevious, CNNLayer layer, ICNNLayer previousLayer,
			int offset, int length) {
		for (int i = offset; i < offset + length; i++) {			
			IConvolutionKernal [] cks = fms[i].getKernals();
			for (int j = 0; j < cks.length; j++) {
				ConvolutionKernal ck = (ConvolutionKernal)cks[j];
				ck.initB = false;
				resetFlagMatrix(ck.getInitDeltaZzs());
			}
			if (CNNUtils.useBN(cfg, fms[i])) {
				CNNUtils.batchNorm(fms[i]);
			} else {
				CNNUtils.deActiveDZzs(fms[i]);
			}
			CNNUtils.deActivateGlobal(this, fms[i]);
		}		
	}
	
//	public void deActivateGW4PartialCurrentLayer(IFeatureMap [] fms, IFeatureMap[] fmsOfPrevious, CNNLayer layer, ICNNLayer previousLayer,
//			int offset, int length) {		
//		for (int i = offset; i < offset + length; i++) {
//			deActivateGlobal(fms[i]);
//		}
//	}
	
	boolean noLock = false;
	public void bp4PartialPreviousLayer(IFeatureMap [] fms, IFeatureMap[] fmsOfPrevious, CNNLayer layer, ICNNLayer previousLayer,
			int offset, int length) {
		noLock = true;
		for (int i = offset; i < offset + length; i++) {
			IFeatureMap ffm = fmsOfPrevious[i];
			for (int j = 0; j < fms.length; j++) {
				IFeatureMap t2fm = fms[j];
				int ckId = t2fm.getfMckIdMap()[i];
				if (ckId >= 0) {
					bpConvolution(previousLayer.getLc(), (ConvolutionKernal) t2fm.getKernals()[ckId], 
							ffm, t2fm,  false);
				}
			}
		}
	}
	
	
	public void visitPartialCNNLayer(final IFeatureMap [] fms, final IFeatureMap[] fmsOfPrevious, final CNNLayer layer, final ICNNLayer previousLayer) {
		int tn = cfg.getThreadsNum();
		if (tn <= 1 || !useParallel) {
			visitPartialCNNLayer(fms, fmsOfPrevious, layer, previousLayer,
				0, fms.length);
		} else {
			threadParallel.runMutipleThreads(fms.length, new PartialCallback() {
				public void runPartial(int offset, int runLen) {
					visitPartialCNNLayer(fms, fmsOfPrevious, layer, previousLayer,
							offset, runLen);
				}
			}, tn);
		}
	}
	public void visitPartialCNNLayer(IFeatureMap [] fms, IFeatureMap[] fmsOfPrevious, CNNLayer layer, ICNNLayer previousLayer,
			int offset, int length) {
		for (int i = offset; i < offset + length; i++) {			
			IConvolutionKernal [] cks = fms[i].getKernals();
			for (int j = 0; j < cks.length; j++) {
				ConvolutionKernal ck = (ConvolutionKernal)cks[j];
				ck.initB = false;
				resetFlagMatrix(ck.getInitDeltaZzs());
			}
		}
		for (int i = offset; i < offset + length; i++) {
			if (CNNUtils.useBN(cfg, fms[i])) {
				CNNUtils.batchNorm(fms[i]);
			}			
			IConvolutionKernal[] cks = fms[i].getKernals();
			for (int j = 0; j < cks.length; j++) {
				IFeatureMap ffm = fmsOfPrevious[cks[j].getFmapOfPreviousLayer()];
				bpConvolution(previousLayer.getLc(), (ConvolutionKernal) cks[j], ffm, fms[i], 
						!CNNUtils.useBN(cfg, fms[i]) && j == 0);
			}
//			deActivateGlobal(fms[i]);
			CNNUtils.deActivateGlobal(this, fms[i]);
		}
	}
	
		
	public void visitResNetLayer(FractalBlock block) {
		FractalBlock [] blocks = block.getFbs();
		IFeatureMap [] fms = block.getFeatureMaps();
		IFeatureMap [] bfms = blocks[blocks.length - 1].getFeatureMaps();
		for (int i = 0; i < fms.length; i++) {
			double [][] dzzs = fms[i].getDeltaZzs();
			double [][] bdzzs = bfms[i].getDeltaZzs();
			for (int j = 0; j < dzzs.length; j++) {
				for (int j2 = 0; j2 < dzzs[j].length; j2++) {
					bdzzs[j][j2] = dzzs[j][j2];
				}
			}
		}
		for (int i = blocks.length - 1; i >= 0; i--) {
			visitFractualBlock(blocks[i]);
		}	
	}
	
	public void visitFractualBlock(FractalBlock block) {
		if (block.isResNet()) {
			visitResNetLayer(block);
		}
		
		CNNLayer dLayer = block.getDirectLayer();
		if (dLayer != null) {
			FractalBlock [] blocks = block.getFbs();
			IFeatureMap [] fms = block.getFeatureMaps();
			IFeatureMap [] dfms = dLayer.getFeatureMaps();
			IFeatureMap [] bfms = blocks[blocks.length - 1].getFeatureMaps();
			for (int i = 0; i < fms.length; i++) {
				double [][] dzzs = fms[i].getDeltaZzs();
				double [][] ddzzs = dfms[i].getDeltaZzs();
				double [][] bdzzs = bfms[i].getDeltaZzs();
				for (int j = 0; j < dzzs.length; j++) {
					for (int j2 = 0; j2 < dzzs[j].length; j2++) {
//						fs[j][j2] = (dfs[j][j2] + bfs[j][j2])/2.0;
//						dzzs[j][j2] = dzzs[j][j2]
//								* fms[i].getAcf().deActivate(fms[i].getzZs()[j][j2]);
						ddzzs[j][j2] = dzzs[j][j2]/2.0;//;
						bdzzs[j][j2] = dzzs[j][j2]/2.0;//;
					}
				}
			}
			visitCNNLayer(dLayer);			
			for (int i = blocks.length - 1; i >= 0; i--) {
				visitFractualBlock(blocks[i]);
			}			
		} else {
			visitCNNLayer(block);
		}		
	}
	
		
	public Object getConvObjs(Object [][] objs, int r, int c, LayerConfigurator lc) {
		int padding = lc.getPadding();
		int mr = r - padding;
		int mc = c - padding;
		if (mr >= 0 && mr <= objs.length - 1 && mc >= 0 && mc <= objs[0].length - 1) {
			return objs[mr][mc];
		}
		return null;
	}
	
	//handle padding issue.
	public double getConvFms(double [][] ffms, int r, int c, LayerConfigurator lc) {
		int padding = lc.getPadding();
		int mr = r - padding;
		int mc = c - padding;
		if (mr >= 0 && mr <= ffms.length - 1 && mc >= 0 && mc <= ffms[0].length - 1) {
			return ffms[mr][mc];
		}
		return 0;
	}
	
	public boolean getConvFms(LayerConfigurator lc, boolean [][] ffms, int r, int c) {
		int padding = lc.getPadding();
		int mr = r - padding;
		int mc = c - padding;
		if (mr >= 0 && mr <= ffms.length - 1 && mc >= 0 && mc <= ffms[0].length - 1) {
			return ffms[mr][mc];
		}
		return false;
	}
	
	public void setConvFms(LayerConfigurator lc, double [][] ffms, int r, int c, double z) {
		int padding = lc.getPadding();
		int mr = r - padding;
		int mc = c - padding;
		if (mr >= 0 && mr <= ffms.length - 1 && mc >= 0 && mc <= ffms[0].length - 1) {
			ffms[mr][mc] = z;
		}
	}
	
	public void setConvFms(LayerConfigurator lc, boolean [][] ffms, int r, int c, boolean z) {
		int padding = lc.getPadding();
		int mr = r - padding;
		int mc = c - padding;
		if (mr >= 0 && mr <= ffms.length - 1 && mc >= 0 && mc <= ffms[0].length - 1) {
			ffms[mr][mc] = z;
		}
	}
	
	public void bpConvolution(LayerConfigurator lc, ConvolutionKernal ck, IFeatureMap ffm, IFeatureMap t2fm, boolean begin) {
		for (int i = 0; i < t2fm.getDeltaZzs().length; i++) {
			for (int j = 0; j < t2fm.getDeltaZzs()[i].length; j++) {	
				if (begin) {
					t2fm.getDeltaZzs()[i][j] = t2fm.getDeltaZzs()[i][j]
							* t2fm.getAcf().deActivate(t2fm.getzZs()[i][j]);
				}
				if (!useGlobalWeight) {
					if (!ck.initB) {
						ck.initB = true;
						ck.deltab = cfg.getM() * ck.deltab
							- cfg.getL() * t2fm.getDeltaZzs()[i][j];
					} else {
						ck.deltab = ck.deltab +
							- cfg.getL() * t2fm.getDeltaZzs()[i][j];
					}
				}
				
				for (int k = 0; k < ck.wWs.length; k++) {
					for (int k2 = 0; k2 < ck.wWs[k].length; k2++) {
//						cs = cs + ffms[i + k][j + k2] * ck.wWs[k][k2];
						int fr = i + k;
						int fc = j + k2;
//						if (!ffm.getInitDeltaZzs()[fr][fc]) {
//							ffm.getInitDeltaZzs()[fr][fc] = true;
//							ffm.getDeltaZzs()[fr][fc] = 
//								t2fm.getDeltaZzs()[i][j] * ck.wWs[k][k2];
//						} else {
//							ffm.getDeltaZzs()[fr][fc] = ffm.getDeltaZzs()[fr][fc] 
//										+ t2fm.getDeltaZzs()[i][j] * ck.wWs[k][k2];
//						}
						Object lockObj = getConvObjs(ffm.getLockObjs(), fr, fc, lc);
						if (!noLock && useParallel && cfg.getThreadsNum() > 1 && lockObj != null) {
							synchronized (lockObj) {
								if (!getConvFms(lc, ffm.getInitDeltaZzs(), fr, fc)) {
									setConvFms(lc, ffm.getInitDeltaZzs(), fr, fc, true);  
									setConvFms(lc, ffm.getDeltaZzs(), fr, fc, t2fm.getDeltaZzs()[i][j] * ck.wWs[k][k2]);
								} else {
									double oz = getConvFms(ffm.getDeltaZzs(), fr, fc, lc);
									setConvFms(lc, ffm.getDeltaZzs(), fr, fc, oz + t2fm.getDeltaZzs()[i][j] * ck.wWs[k][k2]);
								}
							}							
						} else {
							if (!getConvFms(lc, ffm.getInitDeltaZzs(), fr, fc)) {
								setConvFms(lc, ffm.getInitDeltaZzs(), fr, fc, true);  
								setConvFms(lc, ffm.getDeltaZzs(), fr, fc, t2fm.getDeltaZzs()[i][j] * ck.wWs[k][k2]);
							} else {
								double oz = getConvFms(ffm.getDeltaZzs(), fr, fc, lc);
								setConvFms(lc, ffm.getDeltaZzs(), fr, fc, oz + t2fm.getDeltaZzs()[i][j] * ck.wWs[k][k2]);
							}
						}
						//
						double fz = getConvFms(ffm.getFeatures(), fr, fc, lc);
						if (!ck.getInitDeltaZzs()[k][k2]) {
							ck.getInitDeltaZzs()[k][k2] = true;
							ck.detalwWs[k][k2] = cfg.getM() * ck.detalwWs[k][k2]
									- cfg.getL() * t2fm.getDeltaZzs()[i][j] * fz;
						} else {
							ck.detalwWs[k][k2] = ck.detalwWs[k][k2] +
									- cfg.getL() * t2fm.getDeltaZzs()[i][j] * fz;
						}
						if (Math.abs(ck.detalwWs[k][k2]) > 10000) {
							System.out.println("Gradient exploding...");
						}
					}
				}				
			}
		}
	}
	
	public void resetPreviousFlagMatrix(ICNNLayer layer) {
		ICNNLayer previousLayer = layer.getPreviousLayer();
		IFeatureMap[] fmsOfPrevious = previousLayer.getFeatureMaps();
		for (int i = 0; i < fmsOfPrevious.length; i++) {
			resetFlagMatrix(fmsOfPrevious[i].getInitDeltaZzs());
		}
	}
	
	public void resetFlagMatrix(boolean[][] fm) {
		for (int i = 0; i < fm.length; i++) {
			for (int j = 0; j < fm[i].length; j++) {
				fm[i][j] = false;
			}
		}
	}

	@Override
	public void visitPoolingLayer(SamplingLayer layer) {
		resetPreviousFlagMatrix(layer);
		IFeatureMap [] fms = layer.getFeatureMaps();
		IFeatureMap [] fmsInLastLayer = layer.getPreviousLayer().getFeatureMaps();
		visitPartialPoolingLayer(fms, fmsInLastLayer);
	}
	
	public void visitPartialPoolingLayer(final IFeatureMap [] fms, final IFeatureMap [] fmsInLastLayer) {
		int tn = cfg.getThreadsNum();
		if (tn <= 1) {
			visitPartialPoolingLayer(fms, fmsInLastLayer, 
				0, fms.length);
		} else {
			threadParallel.runMutipleThreads(fms.length, new PartialCallback() {
				public void runPartial(int offset, int runLen) {
					visitPartialPoolingLayer(fms, fmsInLastLayer, 
							offset, runLen);
				}
			}, tn);
		}
	}
	
	public void visitPartialPoolingLayer(IFeatureMap [] fms, IFeatureMap [] fmsInLastLayer, 
			int offset, int length) {
		for (int i = offset; i < offset + length; i++) {
			IConvolutionKernal [] cks = fms[i].getKernals();
			//in most cases, the ck is 1 - 1
			for (int j = 0; j < cks.length; j++) {
				SubSamplingKernal ssk = (SubSamplingKernal)cks[j];
				ssk.initwW = false;
				ssk.initB = false;
				IFeatureMap ffm = fmsInLastLayer[ssk.fmapOfPreviousLayer];
				bpSampling(ssk, ffm, fms[i], fms[i].getAcf(), j == 0);
			}
//			deActivateGlobal(fms[i]);
			CNNUtils.deActivateGlobal(this, fms[i]);
		}
	}
	
	boolean useAvgPooling = false;
	public double getPoolingRate(SubSamplingKernal ck) {
		if (!useAvgPooling) {
			return 1;
		} else {
			return 1.0/(double)(ck.ckRows * ck.ckColumns);
		}
	}
	
	public void bpSampling(SubSamplingKernal ck, IFeatureMap ffm, IFeatureMap t2fm, IActivationFunction acf, 
			boolean begin) {
		for (int i = 0; i < t2fm.getDeltaZzs().length; i++) {
			for (int j = 0; j < t2fm.getDeltaZzs()[i].length; j++) {
				if (begin) {
					t2fm.getDeltaZzs()[i][j] = t2fm.getDeltaZzs()[i][j] 
						* acf.deActivate(t2fm.getzZs()[i][j]);
				}				
				if (!ck.initB) {
					ck.initB = true;
					ck.deltab = cfg.getM() * ck.deltab - cfg.getL() * t2fm.getDeltaZzs()[i][j];
				} else {
					ck.deltab = ck.deltab - cfg.getL() * t2fm.getDeltaZzs()[i][j];
				}
				double cs = 0;
				for (int j2 = 0; j2 < ck.ckRows; j2++) {
					for (int k = 0; k < ck.ckColumns; k++) {
//						cs = cs + ffms[i * ck.ckRows + j2]
//								[j * ck.ckColumns + k] * ck.wW;
						int fr = i * ck.ckRows + j2;
						int fc = j * ck.ckColumns + k;
						/*auto padding
						 * **/
						if (fr >= ffm.getFeatures().length || fc >= ffm.getFeatures()[0].length) {
							continue;
						}/*auto padding
						 * **/
						if (!ffm.getInitDeltaZzs()[fr][fc]) {
							ffm.getInitDeltaZzs()[fr][fc] = true;							
							ffm.getDeltaZzs()[fr][fc] = t2fm.getDeltaZzs()[i][j] *
									getPoolingRate(ck) * ck.wW;
						} else {
							ffm.getDeltaZzs()[fr][fc] = ffm.getDeltaZzs()[fr][fc] + 
									t2fm.getDeltaZzs()[i][j] * getPoolingRate(ck) * ck.wW;
						}
						
						if (!ck.initwW) {
							ck.initwW = true;
							ck.deltawW = cfg.getM() * ck.deltawW 
									- cfg.getL() * t2fm.getDeltaZzs()[i][j] * ffm.getFeatures()[fr][fc];
						} else {
							ck.deltawW = ck.deltawW 
									- cfg.getL() * t2fm.getDeltaZzs()[i][j] * ffm.getFeatures()[fr][fc];
						}
					}
				}				
			}
		}
	}
	
	public void cfgCf(CNNLayer2ANNAdapter layer) {
		if (layer.layerImpV2.getCostFunction() == null) {
			layer.layerImpV2.setCostFunction(softMax);
		}
	}

	InputParameters inparams = new InputParameters();
	@Override
	public void visitANNLayer(CNNLayer2ANNAdapter layer) {
		inparams.setAlpha(cfg.getL());
		inparams.setLamda(0);
		inparams.setM(cfg.getM());
		cfgCf(layer);
		layer.layerImpV2.backPropagation(new double[][] {target}, inparams);
		if (layerIndex == cfg.getLayers().length - 1) {
//			softMax.setLayer(layer.layerImpV2);
//			softMax.setTarget(target);
//			softMax.caculateCostError();
			
//			ICostFunction cf = getCf(layer);
//			cf.setLayer(layer.layerImpV2);
//			cf.setTarget(target);
//			cf.caculateCostError();
		}
		if (!(layer.previousLayer instanceof CNNLayer2ANNAdapter)) {
			List<INeuroUnit> list = layer.layerImpV2.getNeuros();
			int cnt = 0;
			IFeatureMap [] fms = layer.getPreviousLayer().getFeatureMaps();
			for (int i = 0; i < fms.length; i++) {
				IFeatureMap fm = fms[i];
				double [][] deltaZzs = fm.getDeltaZzs();
				for (int j = 0; j < deltaZzs.length; j++) {
					for (int j2 = 0; j2 < deltaZzs[j].length; j2++) {
						NeuroUnitImp nu = (NeuroUnitImp) layer.layerImpV2.getNeuros().get(cnt++);
						deltaZzs[j][j2] = nu.getDeltaZ()[0];
					}
				}
			}
		}
	}

	

}
