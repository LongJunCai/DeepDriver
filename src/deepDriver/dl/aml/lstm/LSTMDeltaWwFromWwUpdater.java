package deepDriver.dl.aml.lstm;

import deepDriver.dl.aml.distribution.ITask;

public class LSTMDeltaWwFromWwUpdater implements IRNNLayerVisitor {
	LSTMConfigurator [] fcfgs;
	LSTMConfigurator t2cfg;
	ITask [] tasks;
	int layerPos = 0;
	
	public LSTMDeltaWwFromWwUpdater() {
		super(); 
	}
	public void mergeDeltawWs(LSTMConfigurator [] fcfgs, LSTMConfigurator t2cfg) {
		this.fcfgs = fcfgs;
		this.t2cfg = t2cfg;
		IRNNLayer [] layers = t2cfg.layers;
		for (int i = 0; i < layers.length; i++) {
			layerPos = i;
			layers[i].updateWw(this);
		}
		//updateWw 
	}
	@Override
	public void updateWw4RNNLayer(RNNLayer layer) { 
		RNNNeuroVo [] rNNNeuroVos = layer.getRNNNeuroVos();
		double length = fcfgs.length;
		for (int j = 0; j < fcfgs.length; j++) {
			LSTMConfigurator fcfg = fcfgs[j];
			RNNNeuroVo [] frNNNeuroVos = fcfg.layers[layerPos].getRNNNeuroVos(); 
			for (int i = 0; i < rNNNeuroVos.length; i++) {
				RNNNeuroVo trnVo = rNNNeuroVos[i];
				RNNNeuroVo frnVo = frNNNeuroVos[i];
				updateWws(j == 0, frnVo.getwWs(), trnVo.getwWs(), 
						trnVo.getDeltaWWs(), length);
			}
		}
		
	}
	@Override
	public void updateWw4RNNLayer(LSTMLayer layer) {
		IBlock [] blocks = layer.getBlocks();
		double length = fcfgs.length;
		for (int k = 0; k < fcfgs.length; k++) {
			LSTMConfigurator fcfg = fcfgs[k];
		LSTMLayer flayer = (LSTMLayer) fcfg.layers[layerPos];
		IBlock [] fblocks = flayer.getBlocks();
		for (int i = 0; i < blocks.length; i++) {
			IBlock block = blocks[i];				
			ICell [] cells = block.getCells();
			IInputGate igate = block.getInputGate();
			IOutputGate ogate = block.getOutPutGate();
			IForgetGate fgate = block.getForgetGate();
			
			IBlock fblock = fblocks[i];
			ICell [] fcells = fblock.getCells();
			IInputGate figate = fblock.getInputGate();
			IOutputGate fogate = fblock.getOutPutGate();
			IForgetGate ffgate = fblock.getForgetGate();
			for (int j = 0; j < cells.length; j++) {
				ICell cell = cells[j];	
				ICell fcell = fcells[j];	
				updateWws(k == 0, fcell.getwWs(), cell.getwWs(),cell.getDeltaWWs(), length);
				updateWws(k == 0, fcell.getLwWs(), cell.getLwWs(), cell.getDeltaLwWs(), length);
			}
			updateWw4IRNNNeuroVo(k == 0, figate, igate, length);
			updateWw4IRNNNeuroVo(k == 0, fogate, ogate, length);
			updateWw4IRNNNeuroVo(k == 0, ffgate, fgate, length);
		
		}

		}	
	} 
	
	public void updateWw4IRNNNeuroVo(boolean isNew, IRNNNeuroVo frnvo,IRNNNeuroVo t2rnvo, double length) {
		updateWws(isNew, frnvo.getwWs(),t2rnvo.getwWs(), t2rnvo.getDeltaWWs(), length);
		updateWws(isNew, frnvo.getLwWs(), t2rnvo.getLwWs(), t2rnvo.getDeltaLwWs(), length);
		updateWws(isNew, frnvo.getRwWs(), t2rnvo.getRwWs(), t2rnvo.getDeltaRwWs(), length);
	}
	
	boolean forceLength = true;
	
	public void updateWws(boolean isNew, double [] fWws, double [] t2Wws, double [] t2DeltaWws, double length) {
		/**
		 * <force length = 1>
		 * ***/
		if (forceLength) {
			length = 1;
		}
		/**
		 *  </force length = 1>
		 * ***/
		if (t2Wws == null) {
			return ;
		}
		for (int i = 0; i < t2DeltaWws.length; i++) {			
			if (isNew) {
				t2DeltaWws[i] = (fWws[i] - t2Wws[i])/length;
			} else {
				t2DeltaWws[i] = t2DeltaWws[i] + (fWws[i] - t2Wws[i])/length;
			}			
		}
	}
	@Override
	public void updateWw4RNNLayer(ProjectionLayer layer) {
		// TODO Auto-generated method stub
		
	}

}
