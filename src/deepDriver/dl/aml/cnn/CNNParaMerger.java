package deepDriver.dl.aml.cnn;

import java.io.Serializable;
import java.util.List;

import deepDriver.dl.aml.ann.INeuroUnit;
import deepDriver.dl.aml.distribution.modelParallel.PartialCallback;
import deepDriver.dl.aml.distribution.modelParallel.ThreadParallel;

public class CNNParaMerger implements ICNNLayerVisitor, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	int layerIndex;
	double [][] wWs;
	boolean copy2CNN;
	int cnt = 0;
	public void merge(ConvolutionNeuroNetwork cnn, double [][] wWs, boolean copy2CNN) {
		CNNConfigurator cfg = cnn.getCfg();
		this.wWs = wWs;
		this.copy2CNN = copy2CNN;
		cnt = 0;
		for (int i = 0; i < cfg.getLayers().length; i++) {
			layerIndex = i;
			cfg.getLayers()[i].accept(this);
		}
		System.out.println("Para num is "+cnt);
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
		IFeatureMap [] fms = layer.getFeatureMaps();
		visitPartialCNNLayer(fms);
	}
	
	ThreadParallel threadParallel = new ThreadParallel();
	
	public void visitPartialCNNLayer(final IFeatureMap [] fms) {
		int tn = -1;
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
	
	int wWIndex = 0;
	public void countCNNLayerSize(IFeatureMap [] fms, int offset, int length) {
		wWIndex = 0;
		
		if (copy2CNN) {
			return;
		} 
		int size = 0;		
		for (int i = offset; i < offset + length; i++) {
			IConvolutionKernal[] cks = fms[i].getKernals();
			if (cks != null) {
				for (int j = 0; j < cks.length; j++) {
					ConvolutionKernal ck = (ConvolutionKernal) (cks[j]);
					size = size + ck.wWs.length * ck.wWs[0].length + 1;
				}
			}			
			size = size + 3;			
		}
		cnt = cnt + size;
		wWs[layerIndex] = new double[size];
	}
	
	public void visitPartialCNNLayer(IFeatureMap [] fms, int offset, int length) {
		countCNNLayerSize(fms, offset, length);
		for (int i = offset; i < offset + length; i++) {
			IConvolutionKernal [] cks = fms[i].getKernals(); 			
			if (cks != null) {
				update(cks);
			}
			updateGlobalWws(fms[i]);
		}
	}
	
	private void visitResNetLayer(FractalBlock block) {
		FractalBlock [] blocks = block.getFbs();
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
			visitCNNLayer(dLayer);			
			for (int i = blocks.length - 1; i >= 0; i--) {
				visitFractualBlock(blocks[i]);
			}			
		} else {
			visitCNNLayer(block);
		}		
	}
	
	

	private void updateGlobalWws(IFeatureMap fms) {
		if (copy2CNN) {  
			fms.setGema(wWs[layerIndex][wWIndex++]);
			fms.setBeta(wWs[layerIndex][wWIndex++]);
			fms.setbB(wWs[layerIndex][wWIndex++]);
		} else {
			wWs[layerIndex][wWIndex++] = fms.getGema();
			wWs[layerIndex][wWIndex++] = fms.getBeta();
			wWs[layerIndex][wWIndex++] = fms.getbB();
		} 
	}
	
	private void update(IConvolutionKernal [] cks) {
		for (int i = 0; i < cks.length; i++) {
			update((ConvolutionKernal)cks[i]);
		}
	}
	
	private void update(ConvolutionKernal ck) {
		if (copy2CNN) {
			ck.b = wWs[layerIndex][wWIndex++]; 
		} else {
			wWs[layerIndex][wWIndex++] = ck.b; 
		}
		
		for (int i = 0; i < ck.wWs.length; i++) {
			for (int j = 0; j < ck.wWs[i].length; j++) {
				if (copy2CNN) {
					ck.wWs[i][j] = wWs[layerIndex][wWIndex++]; 
				} else {
					wWs[layerIndex][wWIndex++] = ck.wWs[i][j]; 
				}
			}
		}
	}
	
	private void updateSSk(SubSamplingKernal ck) {
		if (copy2CNN) {
			ck.b = wWs[layerIndex][wWIndex++];
			ck.wW = wWs[layerIndex][wWIndex++];
		} else {
			wWs[layerIndex][wWIndex++] = ck.b;
			wWs[layerIndex][wWIndex++] = ck.wW; 
		}		
	}

	@Override
	public void visitPoolingLayer(SamplingLayer layer) {
		IFeatureMap [] fms = layer.getFeatureMaps();
		visitPartialPoolingLayer(fms);
	}
	
	public void visitPartialPoolingLayer(final IFeatureMap [] fms) {
		int tn = 1;
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
	
	public void countSubSamplingLayerSize(IFeatureMap [] fms, int offset, int length) {
		wWIndex = 0;
		if (copy2CNN) {
			return;
		} 
		int size = 0;
		for (int i = offset; i < offset + length; i++) {
			IConvolutionKernal [] cks = fms[i].getKernals();
			for (int j = 0; j < cks.length; j++) {
				size = size + 2;
			}
			size = size + 3;
		} 	
		cnt = cnt + size;
		wWs[layerIndex] = new double[size];
	}
	
	public void visitPartialPoolingLayer(IFeatureMap [] fms, int offset, int length) {
		countSubSamplingLayerSize(fms, offset, length);
		for (int i = offset; i < offset + length; i++) {
			IConvolutionKernal [] cks = fms[i].getKernals();
			for (int j = 0; j < cks.length; j++) {
				updateSSk((SubSamplingKernal)cks[j]);
			}
			updateGlobalWws(fms[i]);
		}
	}
	
	public void countAnnLayerSize(CNNLayer2ANNAdapter layer) {
		wWIndex = 0;
		if (copy2CNN) {
			return;
		} 
		List<INeuroUnit> list = layer.getLayerImpV2().getNeuros();
		if (list.get(0).getThetas() == null) {
			return;
		}
		int size = list.size() * list.get(0).getThetas().length ;	
		cnt = cnt + size;
		wWs[layerIndex] = new double[size];
	}

	@Override
	public void visitANNLayer(CNNLayer2ANNAdapter layer) {
		countAnnLayerSize(layer);
		List<INeuroUnit> list = layer.getLayerImpV2().getNeuros();
		for (int i = 0; i < list.size(); i++) {
			INeuroUnit nu = list.get(i);
			double [] thetas = nu.getThetas();
			if (thetas == null) {
				return;
			}
			for (int j = 0; j < thetas.length; j++) {
				if (copy2CNN) {
					thetas[j] = wWs[layerIndex][wWIndex++];
				} else {
					wWs[layerIndex][wWIndex++] = thetas[j];
				}
			}
		}
	}

}
