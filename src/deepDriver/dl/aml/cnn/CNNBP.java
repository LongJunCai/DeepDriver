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
			if (fb.getDirectLayer() != null) {
				return true;
			}			
		}
		return false;
	}
	
//	boolean useBN = false;

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
			if (useBN(fms[i])) {
				batchNorm(fms[i]);
			} else {
				deActiveDZzs(fms[i]);
			}
			deActivateGlobal(fms[i]);
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
			if (useBN(fms[i])) {
				batchNorm(fms[i]);
			}			
			IConvolutionKernal[] cks = fms[i].getKernals();
			for (int j = 0; j < cks.length; j++) {
				IFeatureMap ffm = fmsOfPrevious[cks[j].getFmapOfPreviousLayer()];
				bpConvolution(previousLayer.getLc(), (ConvolutionKernal) cks[j], ffm, fms[i], 
						!useBN(fms[i]) && j == 0);
			}
			deActivateGlobal(fms[i]);
		}
	}
	
	public boolean useBN(IFeatureMap t2fm) {
		if (cfg.isUseBN()) {
			if (t2fm.getFeatures().length * t2fm.getFeatures()[0].length > 1) {
				return true;
			}
		}
		return false;
	}
	
	public void deActiveDZzs(IFeatureMap t2fm) {
		double [][] dy = t2fm.getDeltaZzs();
		for (int i = 0; i < dy.length; i++) {
			for (int j = 0; j < dy[i].length; j++) {
				t2fm.getDeltaZzs()[i][j] = t2fm.getDeltaZzs()[i][j]
						* t2fm.getAcf().deActivate(t2fm.getzZs()[i][j]);
			}
		}
	}
	
	public void batchNorm(IFeatureMap t2fm) {
		double [][] y = t2fm.getzZs();
		double [][] xi = t2fm.getoZzs();
		double [][] dy = t2fm.getDeltaZzs();
		
		double [][] dxm = new double[y.length][y[0].length];
		double [][] dxi = new double[xi.length][xi[0].length]; 
		double dvar = 0;
		double du = 0;
		double dgamma = 0;
		double dbeta = 0;
		
		double du1 = 0;
		double du2 = 0;
		double pq = (double)(y.length * y[0].length);
		for (int i = 0; i < dy.length; i++) {
			for (int j = 0; j < dy[i].length; j++) {
				t2fm.getDeltaZzs()[i][j] = t2fm.getDeltaZzs()[i][j]
						* t2fm.getAcf().deActivate(t2fm.getzZs()[i][j]);
				
				dxm[i][j] = dy[i][j] * t2fm.getGema();
				dvar = dvar + dxm[i][j] * (xi[i][j] - t2fm.getU()) 
						* (-0.5) * Math.pow((t2fm.getVar2() + t2fm.getE()), -1.5);
				du1 = du1 - dxm[i][j]/Math.sqrt(t2fm.getVar2() + t2fm.getE());
				du2 = du2 - 2 * (xi[i][j] - t2fm.getU());
				
				double xm = (xi[i][j] - t2fm.getU())/Math.sqrt(t2fm.getVar2() + t2fm.getE());
				if (i == 0 && j == 0) {
					dgamma = cfg.getM() * t2fm.getDgamma() - cfg.getL() * dy[i][j] * xm;
					dbeta = cfg.getM() * t2fm.getDbeta() - cfg.getL() * dy[i][j];
				} else {
					dgamma = dgamma - cfg.getL() * dy[i][j] *  xm;
					dbeta = dbeta - cfg.getL() * dy[i][j];
				}				
			}
		}
		du = du1 + dvar * du2/pq;
		t2fm.setDgamma(dgamma);
		t2fm.setDbeta(dbeta);
		
		for (int i = 0; i < dxi.length; i++) {
			for (int j = 0; j < dxi[0].length; j++) {
				dxi[i][j] = dxm[i][j]/Math.sqrt(t2fm.getVar2() + t2fm.getE())
						+ dvar * 2 * (xi[i][j] - t2fm.getU())/pq + du/pq;
				dy[i][j] = dxi[i][j];
			}
		}
	}
	
	public void visitFractualBlock(FractalBlock block) {
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
	
	public void deActivateGlobal(IFeatureMap fm) {
		if (useBN(fm) || !useGlobalWeight) {
			return;
		} 
		double [][] dzZ = fm.getDeltaZzs();
		fm.setInitBb(false);
		for (int i = 0; i < dzZ.length; i++) {
			for (int j = 0; j < dzZ[i].length; j++) {
				if (!fm.isInitBb()) {
					fm.setInitBb(true);
					fm.setDeltaBb(fm.getDeltaBb() * cfg.getM()
							- cfg.getL() * dzZ[i][j]);
				} else {
					fm.setDeltaBb(fm.getDeltaBb()
							- cfg.getL() * dzZ[i][j]);
				}				
			}
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
			deActivateGlobal(fms[i]);
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
