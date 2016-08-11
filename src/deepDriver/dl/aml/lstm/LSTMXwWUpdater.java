package deepDriver.dl.aml.lstm;

//a visitor pattern should be designed, but due to the cost issue
//will refine this later....
public class LSTMXwWUpdater implements IRNNLayerVisitor {
	
	LSTMConfigurator fcfg;
	LSTMConfigurator t2cfg;
	int layerPos = 0;
	boolean check = false;
	
//	boolean updateWw = false;
	boolean first = true;
	
	public LSTMXwWUpdater() {
		this(false);
	}
	
	public LSTMXwWUpdater(boolean check) {
		super();
		this.check = check;
//		this.updateWw = updateWw;
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
//					updateWws(frnVo.getxWWs(), trnVo.getxWWs()); 
					updateIRNNNeuroVo(frnVo, trnVo); 
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
//				updateWws(fcell.getwWs(), cell.getwWs()); 	
				updateIRNNNeuroVo(fcell, cell); 
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
		updateWws(frnv.getxWWs(), t2rnv.getxWWs());
		updateWws(frnv.getxLwWs(), t2rnv.getxLwWs());
		updateWws(frnv.getxRwWs(), t2rnv.getxRwWs());		
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
				if (resetDeltaWw) {
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
