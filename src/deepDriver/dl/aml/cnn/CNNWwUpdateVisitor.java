package deepDriver.dl.aml.cnn;

import deepDriver.dl.aml.distribution.modelParallel.PartialCallback;
import deepDriver.dl.aml.distribution.modelParallel.ThreadParallel;

public class CNNWwUpdateVisitor implements ICNNLayerVisitor {
	
	CNNBP bp;

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
		visitPartialCNNLayer(fms);
	}
	
	ThreadParallel threadParallel = new ThreadParallel();
	
	public void visitPartialCNNLayer(final IFeatureMap [] fms) {
		int tn = bp.cfg.getThreadsNum();
		if (tn <= 1) {
			visitPartialCNNLayer(fms, 0, fms.length);
		} else {
			threadParallel.runMutipleThreads(fms.length, new PartialCallback() {
				public void runPartial(int offset, int runLen) {
					visitPartialCNNLayer(fms, offset, runLen);
				}
			}, tn);
		}
	}
	
	public void visitPartialCNNLayer(IFeatureMap [] fms, int offset, int length) {
		for (int i = offset; i < offset + length; i++) {
			IConvolutionKernal [] cks = fms[i].getKernals();
			if (cks != null) {
				update(cks);
			}
			updateGlobalWws(fms[i]);
		}
	}
	
	public void visitFractualBlock(FractalBlock block) {
		CNNLayer dLayer = block.getDirectLayer();
		if (dLayer != null) {
			FractalBlock [] blocks = block.getFbs();
			visitCNNLayer(dLayer);			
			for (int i = blocks.length - 1; i >= 0; i--) {
				visitFractualBlock(blocks[i]);
			}			
		} else {
			visitCNNLayer(block);
		}		
	}
	
	private void updateGlobalWws(IFeatureMap fms) {
		fms.setGema(fms.getGema() + fms.getDgamma());
		fms.setBeta(fms.getBeta() + fms.getDbeta());
		if (!bp.useGlobalWeight) {
			return;
		}
		fms.setbB(fms.getbB() + fms.getDeltaBb());
	}
	
	private void update(IConvolutionKernal [] cks) {
		for (int i = 0; i < cks.length; i++) {
			update((ConvolutionKernal)cks[i]);
		}
	}
	
	private void update(ConvolutionKernal ck) {
		ck.b = ck.b + ck.deltab;
		for (int i = 0; i < ck.wWs.length; i++) {
			for (int j = 0; j < ck.wWs[i].length; j++) {
				ck.wWs[i][j] = ck.wWs[i][j] + ck.detalwWs[i][j];
			}
		}
	}
	
	private void updateSSk(SubSamplingKernal ck) {
		ck.b = ck.b + ck.deltab;
		ck.wW = ck.wW + ck.deltawW;
	}

	@Override
	public void visitPoolingLayer(SamplingLayer layer) {
		IFeatureMap [] fms = layer.getFeatureMaps();
		visitPartialPoolingLayer(fms);
	}
	
	public void visitPartialPoolingLayer(final IFeatureMap [] fms) {
		int tn = bp.cfg.getThreadsNum();
		if (tn <= 1) {
			visitPartialPoolingLayer(fms, 0, fms.length);
		} else {
			threadParallel.runMutipleThreads(fms.length, new PartialCallback() {
				public void runPartial(int offset, int runLen) {
					visitPartialPoolingLayer(fms, offset, runLen);
				}
			}, tn);
		}
	}
	
	public void visitPartialPoolingLayer(IFeatureMap [] fms, int offset, int length) {
		for (int i = offset; i < offset + length; i++) {
			IConvolutionKernal [] cks = fms[i].getKernals();
			for (int j = 0; j < cks.length; j++) {
				updateSSk((SubSamplingKernal)cks[j]);
			}
			updateGlobalWws(fms[i]);
		}
	}
	

	@Override
	public void visitANNLayer(CNNLayer2ANNAdapter layer) {
		layer.layerImpV2.updateNeuros();
	}

}
