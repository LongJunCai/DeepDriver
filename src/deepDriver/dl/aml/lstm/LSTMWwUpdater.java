package deepDriver.dl.aml.lstm;

public class LSTMWwUpdater implements IRNNLayerVisitor {
	
	LSTMConfigurator fcfg;
	LSTMConfigurator t2cfg;
	int layerPos = 0;
	boolean check = false;
	
	boolean updateWw = false;
	boolean first = true;
	
	public LSTMWwUpdater() {
		this(false, true);
	}
	
	public LSTMWwUpdater(boolean check, boolean updateWw) {
		super();
		this.check = check;
		this.updateWw = updateWw;
	}

	public void updatewWs(LSTMConfigurator fcfg, LSTMConfigurator t2cfg) {
		first = true;
		this.fcfg = fcfg;
		this.t2cfg = t2cfg;
		IRNNLayer [] layers = t2cfg.layers;
		for (int i = 0; i < layers.length; i++) {
			layerPos = i;
			layers[i].updateWw(this);
		}
		if (check && first) {
			System.out.println("NO DIFF FOUND");
		}
	} 
	
	public void updateWw4RNNLayer(RNNLayer layer) {
		RNNNeuroVo [] rNNNeuroVos = layer.getRNNNeuroVos();
		RNNNeuroVo [] frNNNeuroVos = fcfg.layers[layerPos].getRNNNeuroVos(); 
			for (int i = 0; i < rNNNeuroVos.length; i++) {
				RNNNeuroVo trnVo = rNNNeuroVos[i];
				RNNNeuroVo frnVo = frNNNeuroVos[i];
				if (trnVo.getwWs() != null) {
					if (updateWw) {
						updateWws(frnVo.getwWs(), trnVo.getwWs());
					} else {
						updateWws(frnVo.getDeltaWWs(), trnVo.getDeltaWWs());
					}
					
				}				
			}
	}
	
	public void updateWw4RNNLayer(LSTMLayer layer) {
		IBlock [] blocks = layer.getBlocks();
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
				if (updateWw) {
					updateWws(fcell.getwWs(), cell.getwWs());
					updateWws(fcell.getLwWs(), cell.getLwWs());
				} else {
					updateWws(fcell.getDeltaWWs(), cell.getDeltaWWs());
					updateWws(fcell.getDeltaLwWs(), cell.getDeltaLwWs());
				}
				
			}
			updateIRNNNeuroVo(figate, igate); 			
			updateIRNNNeuroVo(fogate, ogate);			
			updateIRNNNeuroVo(ffgate, fgate);  
			
		}
			
	}
	
	boolean resetDeltaWw = false;	
	
	public boolean isResetDeltaWw() {
		return resetDeltaWw;
	}

	public void setResetDeltaWw(boolean resetDeltaWw) {
		this.resetDeltaWw = resetDeltaWw;
	}

	public void updateIRNNNeuroVo(IRNNNeuroVo frnv, IRNNNeuroVo t2rnv) {
		if (updateWw) {
			updateWws(frnv.getwWs(), t2rnv.getwWs());
			updateWws(frnv.getLwWs(), t2rnv.getLwWs());
			updateWws(frnv.getRwWs(), t2rnv.getRwWs());
		} else {
			updateWws(frnv.getDeltaWWs(), t2rnv.getDeltaWWs());
			updateWws(frnv.getDeltaLwWs(), t2rnv.getDeltaLwWs());
			updateWws(frnv.getDeltaRwWs(), t2rnv.getDeltaRwWs());
		}
	}
	
	public void updateWws(double [] fWws, double [] t2Wws) {		
		if (t2Wws == null) {
			return ;
		}
		for (int i = 0; i < t2Wws.length; i++) {
			if (check) {
				if (t2Wws[i] != fWws[i]) {
					if (first) {
						System.out.println("DIFF is found");
						first = false;
					}					
				}
			} else {
				if (!updateWw && resetDeltaWw) {
					t2Wws[i] = 0;
				} else {
					t2Wws[i] = fWws[i];
				}				
			}			
		}
	}

	@Override
	public void updateWw4RNNLayer(ProjectionLayer layer) {
		// TODO Auto-generated method stub
		
	}

}
