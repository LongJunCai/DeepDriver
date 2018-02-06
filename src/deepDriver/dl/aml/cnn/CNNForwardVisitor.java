package deepDriver.dl.aml.cnn;


import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.distribution.modelParallel.PartialCallback;
import deepDriver.dl.aml.distribution.modelParallel.ThreadParallel;
import deepDriver.dl.aml.math.BlasMathFunction;
import deepDriver.dl.aml.math.MathUtil;

public class CNNForwardVisitor implements ICNNLayerVisitor {

	protected CNNBP bp;
	BlasCNNFdVisitor blasFd;
	@Override
	public void visitCNNLayer(CNNLayer layer) {
		/***visit block
		 * **/
		if (bp.useVisitFractalBlock(layer)) {
			FractalBlock block = (FractalBlock) layer;
			visitFractualBlock(block);
			return;
		}
		/***visit block
		 * **/
		IFeatureMap [] fms = layer.getFeatureMaps();
		if (bp.layerIndex == 0) {
			for (int i = 0; i < fms.length; i++) {
				IDataMatrix dm = bp.dataMatrix[i];
				fms[i].initData(dm);//adaptive to the size;
//				double [][] fs = fms[i].getFeatures();
//				for (int j = 0; j < fs.length; j++) {
//					for (int j2 = 0; j2 < fs[i].length; j2++) {
//						fs[j][j2] = dm.getMatrix()[j][j2];
//					}
//				}
			}
		} else {
			/***Use blas to speed up***/
			if (bp.useBlas()) {
				if (this.blasFd == null) {
					blasFd = new BlasCNNFdVisitor(bp);
					MathUtil.setThreadCnt(bp.cfg.getThreadsNum());
//					MathUtil.registerMf(new BlasMathFunction());
				}
				blasFd.visitCNNLayer(layer);
				return;
			}
			/***Use blas to speed up***/
			IFeatureMap [] fmsInLastLayer = layer.getPreviousLayer().getFeatureMaps();			
			visitPartialCNNLayer(fms, fmsInLastLayer, layer);
		}
	}
	
	public void visitPartialCNNLayer(final IFeatureMap [] fms, final IFeatureMap [] fmsInLastLayer, final CNNLayer layer) {
		int tn = bp.cfg.getThreadsNum();
		if (tn <= 1) {
			visitPartialCNNLayer(fms, fmsInLastLayer, layer,
				0, fms.length);
		} else {
			threadParallel.runMutipleThreads(fms.length, new PartialCallback() {
				public void runPartial(int offset, int runLen) {
					visitPartialCNNLayer(fms, fmsInLastLayer, layer,
							offset, runLen);
				}
			}, tn);
		}		
	}
	
	ThreadParallel threadParallel = new ThreadParallel();
	
	public void visitPartialCNNLayer(IFeatureMap [] fms, IFeatureMap [] fmsInLastLayer, CNNLayer layer,
			int offset, int length) {
		for (int i = offset; i < offset + length; i++) {
				//<adaptive the feature map>
				if (layer.getLc().isFMAdaptive()) {
					fms[i].resizeFeatures();
				}
				//<adaptive the feature map>
				IConvolutionKernal[] cks = fms[i].getKernals();
				for (int j = 0; j < cks.length; j++) {
					IFeatureMap ffm = fmsInLastLayer[cks[j].getFmapOfPreviousLayer()];
					convolution(layer.getPreviousLayer().getLc(), (ConvolutionKernal)cks[j], ffm.getFeatures(), (FeatureMap)fms[i], j == 0);
				}
				CNNUtils.activateConvZzs(bp, bp.cfg, fms[i]);
			}
	}
	
	public void visitResNetLayer(FractalBlock block) {
		FractalBlock [] blocks = block.getFbs();
		for (int i = 0; i < blocks.length; i++) {
			visitFractualBlock(blocks[i]);
		}
		IFeatureMap [] prevFms = block.getPreviousLayer().getFeatureMaps();
		
		IFeatureMap [] fms = block.getFeatureMaps();
		IFeatureMap [] bfms = blocks[blocks.length - 1].getFeatureMaps();
		for (int i = 0; i < fms.length; i++) {
			double [][] fs = fms[i].getFeatures();
			double [][] pfs = prevFms[(int)((double)i * (double)prevFms.length/
					(double)fms.length)].getFeatures();
			double [][] bfs = bfms[i].getFeatures();
			for (int j = 0; j < fs.length; j++) {
				for (int j2 = 0; j2 < fs[j].length; j2++) {
					fs[j][j2] = (bfs[j][j2] + pfs[(int)((double)j * (double)pfs.length/(double)fs.length)]
							[(int)((double)j2 * (double)pfs[j].length/(double)fs[j].length)]);//);//;j2
				}
			}
		}
	}
	
	
	
	public void visitFractualBlock(FractalBlock block) {
		if (block.isResNet()) {
			visitResNetLayer(block);
		}
		CNNLayer dLayer = block.getDirectLayer();
		if (dLayer != null) {
			visitCNNLayer(dLayer);
			
			FractalBlock [] blocks = block.getFbs();
			for (int i = 0; i < blocks.length; i++) {
				visitFractualBlock(blocks[i]);
			}
			
			IFeatureMap [] fms = block.getFeatureMaps();
			IFeatureMap [] dfms = dLayer.getFeatureMaps();
			IFeatureMap [] bfms = blocks[blocks.length - 1].getFeatureMaps();
			for (int i = 0; i < fms.length; i++) {
				double [][] fs = fms[i].getFeatures();
				double [][] dfs = dfms[i].getFeatures();
				double [][] bfs = bfms[i].getFeatures();
				for (int j = 0; j < fs.length; j++) {
					for (int j2 = 0; j2 < fs[j].length; j2++) {
						fs[j][j2] = (bfs[j][j2] + dfs[j][j2])/2.0;//);//;
					}
				}
			}
		} else {
			visitCNNLayer(block);
		}		
	}	
			
	
	public void convolution(LayerConfigurator lc, ConvolutionKernal ck, double [][] ffms, FeatureMap t2fm, boolean begin) {
		for (int i = 0; i < t2fm.getFeatures().length; i++) {
			for (int j = 0; j < t2fm.getFeatures()[i].length; j++) {				
				double cs = 0;
				for (int k = 0; k < ck.wWs.length; k++) {
					for (int k2 = 0; k2 < ck.wWs[k].length; k2++) {
//						cs = cs + ffms[i + k][j + k2] * ck.wWs[k][k2];
						cs = cs + bp.getConvFms(ffms, i + k, j + k2, lc) * ck.wWs[k][k2];
					}
				}
				if (!bp.useGlobalWeight) {
					cs = cs + ck.b;
				}				
//				double acs = t2fm.acf.activate(cs);
				//no activation here.
				if (begin) {
					t2fm.getzZs()[i][j] = cs;
//					t2fm.getFeatures()[i][j] = acs;
				} else {
					t2fm.getzZs()[i][j] = t2fm.getzZs()[i][j] + cs;
//					t2fm.getFeatures()[i][j] = t2fm.getFeatures()[i][j] + acs;
				}
			}
		}
	}

	@Override
	public void visitPoolingLayer(SamplingLayer layer) {
		IFeatureMap [] fms = layer.getFeatureMaps();
		IFeatureMap [] fmsInLastLayer = layer.getPreviousLayer().getFeatureMaps();
		/***Use blas to speed up
		if (bp.useBlas()) {
			if (this.blasFd == null) {
				blasFd = new BlasCNNFdVisitor(bp);
				MathUtil.setThreadCnt(bp.cfg.getThreadsNum());
			}
			blasFd.visitPoolingLayer(layer);
			return;
		}***/
		/***Use blas to speed up***/
		visitPartialPoolingLayer(fms, fmsInLastLayer, layer);
	}
	
	public void visitPartialPoolingLayer(final IFeatureMap [] fms, final IFeatureMap [] fmsInLastLayer, final SamplingLayer layer) {
		int tn = bp.cfg.getThreadsNum();
		if (tn <= 1) {
			visitPartialPoolingLayer(fms, fmsInLastLayer, layer,
				0, fms.length);
		} else {
			threadParallel.runMutipleThreads(fms.length, new PartialCallback() {
				public void runPartial(int offset, int runLen) {
					visitPartialPoolingLayer(fms, fmsInLastLayer, layer,
							offset, runLen);
				}
			}, tn);
		}
	}
	
	public void visitPartialPoolingLayer(IFeatureMap [] fms, IFeatureMap [] fmsInLastLayer, SamplingLayer layer, 
			int offset, int length) {
		for (int i = offset; i < offset + length; i++) {
			IConvolutionKernal[] cks = fms[i].getKernals();
			//in most cases, the ck is 1 - 1
			for (int j = 0; j < cks.length; j++) {
				SubSamplingKernal ssk = (SubSamplingKernal)cks[j];
				IFeatureMap ffm = fmsInLastLayer[cks[j].getFmapOfPreviousLayer()]; 
				//<adapt to ck>
				if (layer.getLc().isCKAdaptive()) {
				    if (ffm.getFeatures()[0].length - fms[i].getFeatures()[0].length + 1 <= 0) {
                        System.out.println("Error ...");
                    }
					ssk.ckRows = ffm.getFeatures().length - fms[i].getFeatures().length + 1;
					ssk.ckColumns = ffm.getFeatures()[0].length - fms[i].getFeatures()[0].length + 1;
				}
				//<adapt to ck>				
				sampling(ssk, ffm.getFeatures(), fms[i], j == 0, fms[i].getAcf());
			}
			CNNUtils.activateZzs(bp, fms[i]);
		}
	}
	
	public void sampling(SubSamplingKernal ck, double [][] ffms, IFeatureMap t2fm, boolean begin, IActivationFunction acf) {
		for (int i = 0; i < t2fm.getFeatures().length; i++) {
			for (int j = 0; j < t2fm.getFeatures()[i].length; j++) {
				double cs = 0;
				for (int j2 = 0; j2 < ck.ckRows; j2++) {
					for (int k = 0; k < ck.ckColumns; k++) {
						int fr = i * ck.ckRows + j2;
						int fc = j * ck.ckColumns + k;
						/*auto padding
						 * **/
						if (fr >= ffms.length || fc >= ffms[0].length) {
							continue;
						}/*auto padding
						 * **/
						cs = cs + ffms[fr][fc] * ck.wW;
					}
				}
				if (!bp.useGlobalWeight) {
					cs = cs + ck.b;
				}				
//				double acs = acf.activate(cs);
				if (begin) {
//					t2fm.getFeatures()[i][j] = acs;
					t2fm.getzZs()[i][j] = bp.getPoolingRate(ck) * cs;
				} else {
//					t2fm.getFeatures()[i][j] = t2fm.getFeatures()[i][j] + acs;
					t2fm.getzZs()[i][j] = t2fm.getzZs()[i][j] + bp.getPoolingRate(ck) * cs;
				}
			}
		}
	}
	
	double [][] input;

	@Override
	public void visitANNLayer(CNNLayer2ANNAdapter layer) {
		if (layer.layerImpV2.getPreviousLayer() == null) {
			//read convolution value here.
			input = new double [][] {
					layer.previousLayer.featureMaps2Vector()};
			layer.layerImpV2.forwardPropagation(input);
		} else {
			bp.cfgCf(layer);
			layer.layerImpV2.forwardPropagation(input);
			if (bp.layerIndex == bp.cfg.getLayers().length - 1) {
				bp.result = layer.layerImpV2.getRs();
				bp.stdError = layer.layerImpV2.getStdError(new double[][] {bp.target});
				/*cf.setLayer(layer.layerImpV2);
				bp.result = cf.activate();
				cf.setTarget(bp.target);
				bp.stdError = cf.caculateStdError();**/
			}
		}
	}

}
