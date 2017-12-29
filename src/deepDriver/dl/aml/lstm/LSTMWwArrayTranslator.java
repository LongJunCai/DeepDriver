package deepDriver.dl.aml.lstm;

import java.io.Serializable;

public class LSTMWwArrayTranslator implements IRNNLayerVisitor, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	LSTMConfigurator fcfg;
	double [][] wWs;
	int layerPos = 0; 
	
	boolean w2cfg = false;  

	public LSTMWwArrayTranslator() {
		super();
	}

	public void update(LSTMConfigurator fcfg, double[][] wWs, boolean a2wWs) { 
		this.fcfg = fcfg;
		this.wWs = wWs;
		this.w2cfg = a2wWs; 
		IRNNLayer [] layers = fcfg.layers;
		for (int i = 0; i < layers.length; i++) {
			layerPos = i;
			layers[i].updateWw(this);
		} 
	} 
	
	int wWIndex = 0;
	public void count4RNNLayer(RNNLayer layer) {
		wWIndex = 0;
		if (w2cfg) {
			return;
		}

		int cnt = 0;
		RNNNeuroVo[] rNNNeuroVos = layer.getRNNNeuroVos();
		for (int i = 0; i < rNNNeuroVos.length; i++) {
			RNNNeuroVo trnVo = rNNNeuroVos[i];
			cnt = cnt + countWws4IRNNNeuroVo(trnVo);
		}
		wWs[layerPos] = new double[cnt];
	}
	
	public int countWws4IRNNNeuroVo(IRNNNeuroVo trnVo) {
		int cnt = 0;
		if (trnVo.getwWs() != null) {
			cnt = cnt + trnVo.getwWs().length;
		}
		if (trnVo.getLwWs() != null) {
			cnt = cnt + trnVo.getLwWs().length;
		}
		if (trnVo.getRwWs() != null) {
			cnt = cnt + trnVo.getRwWs().length;
		}
		return cnt;
	}
	
	public void updateWws4IRNNNeuroVo(IRNNNeuroVo trnVo) {
		if (trnVo.getwWs() != null) {
			updateWws(trnVo.getwWs());
		}
		if (trnVo.getLwWs() != null) {
			updateWws(trnVo.getLwWs());
		}
		if (trnVo.getRwWs() != null) {
			updateWws(trnVo.getRwWs());
		}
	}
	
	public void updateWw4RNNLayer(RNNLayer layer) {
		count4RNNLayer(layer);
		
		RNNNeuroVo[] rNNNeuroVos = layer.getRNNNeuroVos();
		for (int i = 0; i < rNNNeuroVos.length; i++) {
			RNNNeuroVo trnVo = rNNNeuroVos[i];
			updateWws4IRNNNeuroVo(trnVo);
		}
	}
	
	public void count4RNNLayer(LSTMLayer layer) {
		wWIndex = 0;
		if (w2cfg) {
			return;
		}

		int cnt = 0;
		IBlock [] blocks = layer.getBlocks();
		for (int i = 0; i < blocks.length; i++) {
			IBlock block = blocks[i];				
			ICell [] cells = block.getCells();
			IInputGate igate = block.getInputGate();
			IOutputGate ogate = block.getOutPutGate();
			IForgetGate fgate = block.getForgetGate();
			  
			for (int j = 0; j < cells.length; j++) {
				ICell cell = cells[j]; 
				cnt = cnt + countWws4IRNNNeuroVo(cell);
			}  
			cnt = cnt + countWws4IRNNNeuroVo(igate);
			cnt = cnt + countWws4IRNNNeuroVo(ogate);
			cnt = cnt + countWws4IRNNNeuroVo(fgate);
		}
		wWs[layerPos] = new double[cnt];	
	}
	
	public void updateWw4RNNLayer(LSTMLayer layer) {
		count4RNNLayer(layer);
		
		IBlock [] blocks = layer.getBlocks();
		for (int i = 0; i < blocks.length; i++) {
			IBlock block = blocks[i];				
			ICell [] cells = block.getCells();
			IInputGate igate = block.getInputGate();
			IOutputGate ogate = block.getOutPutGate();
			IForgetGate fgate = block.getForgetGate();
			  
			for (int j = 0; j < cells.length; j++) {
				ICell cell = cells[j]; 
				updateWws4IRNNNeuroVo(cell);
			}  
			updateWws4IRNNNeuroVo(igate);
			updateWws4IRNNNeuroVo(ogate);
			updateWws4IRNNNeuroVo(fgate); 
		}
			
	}
	
	boolean resetDeltaWw = false;	
	
	public boolean isResetDeltaWw() {
		return resetDeltaWw;
	}

	public void setResetDeltaWw(boolean resetDeltaWw) {
		this.resetDeltaWw = resetDeltaWw;
	}

	public void updateWws(double [] cfgWws) {		
		if (cfgWws == null) {
			return ;
		}
		for (int i = 0; i < cfgWws.length; i++) {
			if (w2cfg) {
				cfgWws[i] = wWs[layerPos][wWIndex++];
			} else {
				wWs[layerPos][wWIndex++] = cfgWws[i];
			}			
		}
	}

	@Override
	public void updateWw4RNNLayer(ProjectionLayer layer) { 
		
	}

}
