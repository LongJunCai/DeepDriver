package deepDriver.dl.aml.lstm;

public class LSTMWwFresher implements IRNNLayerVisitor {
	
	LSTMConfigurator t2cfg;
	int layerPos = 0;
	public void freshwWs(LSTMConfigurator t2cfg) { 
		this.t2cfg = t2cfg;
		IRNNLayer [] layers = t2cfg.layers;
		for (int i = 0; i < layers.length; i++) {
			layerPos = i;
			layers[i].updateWw(this);
		}
	} 
	
	public void updateWw4RNNLayer(RNNLayer layer) {
		RNNNeuroVo [] rNNNeuroVos = layer.getRNNNeuroVos();
			for (int i = 0; i < rNNNeuroVos.length; i++) {
				RNNNeuroVo trnVo = rNNNeuroVos[i]; 
				updateWws(trnVo.getwWs(), trnVo.getDeltaWWs());
			}
	}
	
	public void updateWw4RNNLayer(LSTMLayer layer) {
		IBlock [] blocks = layer.getBlocks();
		for (int i = 0; i < blocks.length; i++) {
			IBlock block = blocks[i];				
			ICell [] cells = block.getCells();
			IInputGate igate = block.getInputGate();
			IOutputGate ogate = block.getOutPutGate();
			IForgetGate fgate = block.getForgetGate();
			 
			for (int j = 0; j < cells.length; j++) {
				ICell cell = cells[j];	 
				updateWws(cell.getwWs(), cell.getDeltaWWs());
				updateWws(cell.getLwWs(), cell.getDeltaLwWs());
			}
			updateWws(igate.getwWs(), igate.getDeltaWWs());
			updateWws(igate.getLwWs(), igate.getDeltaLwWs());
			updateWws(igate.getRwWs(), igate.getDeltaRwWs());
			
			updateWws(ogate.getwWs(), ogate.getDeltaWWs());
			updateWws(ogate.getLwWs(), ogate.getDeltaLwWs());
			updateWws(ogate.getRwWs(), ogate.getDeltaRwWs());
			
			updateWws(fgate.getwWs(), fgate.getDeltaWWs());
			updateWws(fgate.getLwWs(), fgate.getDeltaLwWs());
			updateWws(fgate.getRwWs(), fgate.getDeltaRwWs());
			
		}
			
	}
	
	public void updateWws(double [] wWs, double [] deltaWws) {
		if (wWs == null) {
			return;
		}
		for (int i = 0; i < wWs.length; i++) {
			wWs[i] = wWs[i] + deltaWws[i];
		}
	}

	@Override
	public void updateWw4RNNLayer(ProjectionLayer layer) {
		// TODO Auto-generated method stub
		
	}

}