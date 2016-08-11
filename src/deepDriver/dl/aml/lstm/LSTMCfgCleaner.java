package deepDriver.dl.aml.lstm;

public class LSTMCfgCleaner implements IRNNLayerVisitor {
	 
	LSTMConfigurator t2cfg; 
	
	public LSTMCfgCleaner() { 
	}
	 
	int layerPos;
	
	public void clean(Seq2SeqLSTMConfigurator [] cfgs) {
		for (int i = 0; i < cfgs.length; i++) {
			clean(cfgs[i]);
		}
	}
	
	public void clean(Seq2SeqLSTMConfigurator seq2SeqLSTMConfigurator) {
		clean(seq2SeqLSTMConfigurator.getQlSTMConfigurator());
		clean(seq2SeqLSTMConfigurator.getAlSTMConfigurator());
		seq2SeqLSTMConfigurator.setAlSTMConfigurator(null);
		seq2SeqLSTMConfigurator.setQlSTMConfigurator(null);
	}
	
	public void gbClean() {
		System.gc();
	}

	public void clean(LSTMConfigurator t2cfg) {
		this.t2cfg = t2cfg;
		IRNNLayer [] layers = t2cfg.layers;
		for (int i = 0; i < layers.length; i++) {
			layerPos = i;
			layers[i].updateWw(this);
		}
		t2cfg.cxtConsumer = null;
		t2cfg.layers = null;
		t2cfg.nna = null;
		t2cfg.setPreCxtProvider(null);
	} 
	
	public void updateWw4RNNLayer(RNNLayer layer) {
		RNNNeuroVo [] rNNNeuroVos = layer.getRNNNeuroVos();
			for (int i = 0; i < rNNNeuroVos.length; i++) {
				RNNNeuroVo trnVo = rNNNeuroVos[i];
				destroy(trnVo);			
			}
		layer.setRNNNeuroVos(null);
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
				destroy(cell);				
			}
			destroy(igate);
			destroy(ogate);
			destroy(fgate); 	
			block.setCells(null);
		}		
		layer.setBlocks(null);
	}
	
	public void destroy(ICell t2rnv) {
		destroy((IRNNNeuroVo)t2rnv);
		t2rnv.setCZz(null);
		t2rnv.setDeltaC(null);
		t2rnv.setDeltaSc(null);
		t2rnv.setSc(null);
	}
	
	public void destroy(IRNNNeuroVo t2rnv) {
		t2rnv.setDeltaLwWs(null);
		t2rnv.setDeltaRwWs(null);
		t2rnv.setDeltaWWs(null);
		t2rnv.setLwWs(null);		
		t2rnv.setRwWs(null);
		t2rnv.setwWs(null);
		destroyNeuroVos(t2rnv);
		t2rnv.setNeuroVos(null);
	}
	
	public void destroyNeuroVos(IRNNNeuroVo t2rnv) {
		SimpleNeuroVo[] neuroVos = t2rnv.getNvTT();
		if (neuroVos == null) {
			return ;
		}
		for (int i = 0; i < neuroVos.length; i++) {
			neuroVos[i] = null;
		}
	}

	@Override
	public void updateWw4RNNLayer(ProjectionLayer layer) {
		// TODO Auto-generated method stub
		
	}
	

}
