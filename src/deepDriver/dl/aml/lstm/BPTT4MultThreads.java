package deepDriver.dl.aml.lstm;


import deepDriver.dl.aml.distribution.modelParallel.PartialCallback;
import deepDriver.dl.aml.distribution.modelParallel.ThreadParallel;
import deepDriver.dl.aml.lstm.imp.Block;

public class BPTT4MultThreads extends BPTT {
	
	ThreadParallel threadParallel;

	public BPTT4MultThreads(LSTMConfigurator cfg) {
		super(cfg);
		threadParallel = new ThreadParallel();
	}
	 
	public void runMutipleThreads(int length, PartialCallback p) {
		int tn = cfg.getThreadsNum();
		threadParallel.runMutipleThreads(length, p, tn);
	}
	
	public void fTT4FeatureAssignment(RNNNeuroVo [] vos, double [] feature) {
		runMutipleThreads(vos.length, new PartialCallback() {
			public void runPartial(int offset, int runLen) {
				fTT4FeatureAssignment(vos, feature, offset, runLen);
			}			
		});	
	}
	
	public void fTT4PartialRNNLayer(final RNNNeuroVo [] vos, final RNNNeuroVo [] previousVos) {
//		fTT4PartialRNNLayer(vos, previousVos, 0, vos.length);
		runMutipleThreads(vos.length, new PartialCallback() {
			public void runPartial(int offset, int runLen) {
				fTT4PartialRNNLayer(vos, previousVos, offset, runLen);
			}			
		});		
	}
	
	public void fTT4PartialLstmLayer(final Block [] blocks, final RNNNeuroVo [] previousVos, final LSTMLayer layer, final int binaryPos, final boolean speedUpLearning) {
//		fTT4PartialLstmLayer(blocks, 0, blocks.length, previousVos, layer, binaryPos, speedUpLearning);
		runMutipleThreads(blocks.length, new PartialCallback() {
			public void runPartial(int offset, int runLen) {
				fTT4PartialLstmLayer(blocks, offset, runLen, previousVos, layer, binaryPos, speedUpLearning);
			}			
		});
	}
	
	public void bpTT4PartialRNNLayerCell(final ICell[] allCells, final LSTMLayer layer) {
//		bpTT4PartialRNNLayerCell(allCells, layer, 0, allCells.length);
		runMutipleThreads(allCells.length, new PartialCallback() {
			public void runPartial(int offset, int runLen) {
				bpTT4PartialRNNLayerCell(allCells, layer, offset, runLen);
			}			
		});
	}
	
	public void bpTT4PartialRNNLayerBlocks(final Block [] blocks, final LSTMLayer layer) {
//		bpTT4PartialRNNLayerBlocks(blocks, layer, 0, blocks.length);
		runMutipleThreads(blocks.length, new PartialCallback() {
			public void runPartial(int offset, int runLen) {
				bpTT4PartialRNNLayerBlocks(blocks, layer, offset, runLen);
			}			
		});
	}
	
	public void updateWw4PartialRNNLayer(final RNNNeuroVo [] rNNNeuroVos) {
//		updateWw4PartialRNNLayer(rNNNeuroVos, 0, rNNNeuroVos.length);
		runMutipleThreads(rNNNeuroVos.length, new PartialCallback() {
			public void runPartial(int offset, int runLen) {
				updateWw4PartialRNNLayer(rNNNeuroVos, offset, runLen);
			}			
		});
	}
	
	public void updateWw4PartialLstmLayer(final LSTMLayer layer, final IBlock [] blocks) {
//		updateWw4PartialRNNLayer(layer, blocks, 0, blocks.length);
		runMutipleThreads(blocks.length, new PartialCallback() {
			public void runPartial(int offset, int runLen) {
				updateWw4PartialRNNLayer(layer, blocks, offset, runLen);
			}			
		});
	}
	
	public void bpttPartialFromNextLayer(final RNNNeuroVo [] vos, final IRNNLayer layer, final boolean useDeActivate) {
//		bpttPartialFromNextLayer(vos, layer, useDeActivate, 0, vos.length);
		runMutipleThreads(vos.length, new PartialCallback() {
			public void runPartial(int offset, int runLen) {
				bpttPartialFromNextLayer(vos, layer, useDeActivate, offset, runLen);
			}			
		});
	}
	
	public void bpttPartialFromNextLayer(final IRNNLayer nextLayer,  final RNNNeuroVo [] vos, final IRNNLayer layer, final boolean useDeActivate, final boolean addtive) {
//		bpttPartialFromNextLayer(nextLayer,  vos, layer, useDeActivate, 0, vos.length, false);
		runMutipleThreads(vos.length, new PartialCallback() {
			public void runPartial(int offset, int runLen) {
				bpttPartialFromNextLayer(nextLayer, vos, layer, useDeActivate, offset, runLen, addtive);
			}			
		});
	}

}
