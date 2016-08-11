package deepDriver.dl.aml.lstm;

import deepDriver.dl.aml.distribution.ITask;

public class LSTMDeltaWwUpdater implements IRNNLayerVisitor {
	LSTMConfigurator [] fcfgs;
	LSTMConfigurator t2cfg;
	ITask [] tasks;
	int layerPos = 0;
	
	double l = 1;
	double m = 0;
	
	public LSTMDeltaWwUpdater() {
		super(); 
	}
	public void mergeDeltawWs(LSTMConfigurator [] fcfgs, 
			LSTMConfigurator t2cfg, double l, double m) {
		this.l = l;
		this.m = m;
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
				updateWws(j == 0, frnVo.getDeltaWWs(), trnVo.getDeltaWWs(), length);
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
				updateWws(k == 0, fcell.getDeltaWWs(), cell.getDeltaWWs(), length);
				updateWws(k == 0, fcell.getDeltaLwWs(), cell.getDeltaLwWs(), length);
			}
			updateWw4IRNNNeuroVo(k == 0, figate, igate, length);
			updateWw4IRNNNeuroVo(k == 0, fogate, ogate, length);
			updateWw4IRNNNeuroVo(k == 0, ffgate, fgate, length);
		
		}

		}	
	} 
	
	public void updateWw4IRNNNeuroVo(boolean isNew, IRNNNeuroVo frnvo,IRNNNeuroVo t2rnvo, double length) {
		updateWws(isNew, frnvo.getDeltaWWs(), t2rnvo.getDeltaWWs(), length);
		updateWws(isNew, frnvo.getDeltaLwWs(), t2rnvo.getDeltaLwWs(), length);
		updateWws(isNew, frnvo.getDeltaRwWs(), t2rnvo.getDeltaRwWs(), length);
	}
	
	boolean forceLength = true;
	
	public void updateWws(boolean isNew, double [] fWws, double [] t2Wws, double length) {
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
		for (int i = 0; i < t2Wws.length; i++) {			
			if (isNew) {
				t2Wws[i] = fWws[i]/length * l + m * t2Wws[i];
			} else {
				t2Wws[i] = t2Wws[i] + fWws[i]/length * l;
			}			
		}
	}
	@Override
	public void updateWw4RNNLayer(ProjectionLayer layer) {
		// TODO Auto-generated method stub
		
	}

}
