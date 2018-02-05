package deepDriver.dl.aml.lstm;

public class BiBPTT extends BPTT4MultThreads { 

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	//assume all the lstm layers are BiLstmLayer
	public BiBPTT(LSTMConfigurator cfg) {
		super(cfg);
	}
	
	int NORMAL = 1;
	int INVERSE = 2;
	int biDirection = NORMAL;
	
	protected double [][] fTT(double [][] sample) {
		this.sample = sample;
		
		for (int j = 0; j < cfg.layers.length; j++) {
				layerPos = j;
				for (int i = 0; i < sample.length; i++) {
					biDirection = NORMAL;
					t = i;//not sure how to handle the bi-directional
					feature = sample[i];
					cfg.layers[j].fTT(this);
				}		
				if (j ==0 || (j == cfg.layers.length - 1 && cfg.isRequireLastRNNLayer())
						|| isProjectLayer(cfg.layers[j])) {
					continue;//RNN Layer has no need to run the bi
				}
				for (int i = sample.length - 1; i >= 0 ; i--) {//reverse order
					biDirection = INVERSE;
					t = sample.length - 1 - i;//it is trick here.
					feature = sample[i];
					cfg.layers[j].fTT(this);
				}
		}
		t = sample.length - 1;
		
		IRNNNeuroVo [] nvs = cfg.layers[cfg.layers.length - 1].getRNNNeuroVos();
		double [][] results = new double[t + 1][nvs.length];
		for (int i = 0; i < results.length; i++) {
			results[i] = new double[nvs.length];
			for (int j = 0; j < results[i].length; j++) {
				results[i][j] = nvs[j].getNvTT()[i].aA;
			}
		}
		return results;
	}
	
	
	public double bptt(double [][] targets) {
		error = 0;
		for (int j = (cfg.layers.length - 1); j >= 0; j--) {
			layerPos = j;
			for (int i = (targets.length - 1); i >= ngram; i--) {
				biDirection = NORMAL;
				t = i;
				target = targets[i];
				if (j == cfg.layers.length - 1 && cfg.isRequireLastRNNLayer()) {
					error = error + caculateError(target,
									cfg.layers[cfg.layers.length - 1]
											.getRNNNeuroVos(), t);
				}
				cfg.layers[j].bpTT(this);
			}
			
			if ((j == cfg.layers.length - 1 && cfg.isRequireLastRNNLayer())
					) {
				continue;
			}
			
			for (int i = ngram; i < (targets.length); i ++) {
				biDirection = INVERSE;
				t = targets.length - 1 - i;
				target = targets[i];
				
				cfg.layers[j].bpTT(this);
			}
		}
		return error;
	}
	
//	public void fTT4RNNLayer(ProjectionLayer layer) {
//		if (layerPos == cfg.layers.length - 1) {
//			BiLstmLayer prev = (BiLstmLayer)cfg.layers[layerPos - 1];
//			prev.reverseOpposite(sample.length);
//			super.fTT4RNNLayer(layer);
//			prev.reverseBackOpposite();
//		} else {
//			super.fTT4RNNLayer(layer);
//		}
//	}


	@Override
	public void fTT4RNNLayer(RNNLayer layer) {
		if (layerPos == cfg.layers.length - 1) {
			BiLstmLayer prev = (BiLstmLayer)cfg.layers[layerPos - 1];
			prev.reverseOpposite(sample.length);
			super.fTT4RNNLayer(layer);
			prev.reverseBackOpposite();
		} else {
			super.fTT4RNNLayer(layer);
		}
	}


	@Override
	public void fTT4RNNLayer(LSTMLayer layer) {
		BiLstmLayer biLayer = (BiLstmLayer) layer;
		IRNNLayer pl = cfg.layers[layerPos - 1];
		if (layerPos - 1 == 0) {
			BiRNNLayer prev = (BiRNNLayer)pl;
			if (biDirection == NORMAL) {	
//				feature = sample[t];
				super.fTT4RNNLayer(biLayer.layer);
			} else {
//				feature = sample[sample.length - 1 - t];
				prev.reverse(sample.length);
				super.fTT4RNNLayer(biLayer.ilayer);
				prev.reverseBack();
			}			
		} else if(isProjectLayer(pl)) {
			BiProjectionLayer2 prev = (BiProjectionLayer2)pl;
			if (biDirection == NORMAL) {	
//				feature = sample[t];
				super.fTT4RNNLayer(biLayer.layer);
			} else {
//				feature = sample[sample.length - 1 - t];
				prev.reverse(sample.length);
				super.fTT4RNNLayer(biLayer.ilayer);
				prev.reverseBack();
			}
		} else {
			BiLstmLayer prev = (BiLstmLayer)cfg.layers[layerPos - 1];
			if (biDirection == NORMAL) {				
				prev.reverseOpposite(sample.length);
				super.fTT4RNNLayer(biLayer.layer);
				prev.reverseBackOpposite();
			} else {
				prev.reverseNormal(sample.length);
				super.fTT4RNNLayer(biLayer.ilayer);
				prev.reverseBackNormal();
			}			
		}
	}


	@Override
	public void bpTT4RNNLayer(RNNLayer layer) {
		//
		if (layerPos - 1 == 0) {
			BiRNNLayer brLayer = (BiRNNLayer) layer;
			if (biDirection == NORMAL) {
				IRNNLayer nLayer = cfg.layers[layerPos + 1];
				if (nLayer instanceof BiLstmLayer) {
					BiLstmLayer nbLayer = (BiLstmLayer) nLayer;
					nLayer = nbLayer.layer;
				}
				bpttPartialFromNextLayer(nLayer, layer.getRNNNeuroVos(), layer, false, false);
			} else {
				brLayer.reverse(sample.length);
				IRNNLayer nLayer = cfg.layers[layerPos + 1];
				if (nLayer instanceof BiLstmLayer) {
					BiLstmLayer nbLayer = (BiLstmLayer) nLayer;
					nLayer = nbLayer.ilayer;
				}
				bpttPartialFromNextLayer(nLayer, layer.getRNNNeuroVos(), layer, false, true);
				brLayer.reverseBack();
			}
		} else {
			super.bpTT4RNNLayer(layer);
		}		
	}

	public void bpTT4RNNLayer(ProjectionLayer layer) {
		BiProjectionLayer2 cLayer = (BiProjectionLayer2) layer;
		IRNNLayer nLayer = cfg.layers[layerPos + 1];		
		if (biDirection == NORMAL) {	
			if (nLayer instanceof BiLstmLayer) {
				BiLstmLayer nbLayer = (BiLstmLayer) nLayer;
				nLayer = nbLayer.layer;
			}
			bpttPartialFromNextLayer(nLayer, layer.getRNNNeuroVos(), layer, false, false);
		} else {
			cLayer.reverse(sample.length);
			if (nLayer instanceof BiLstmLayer) {
				BiLstmLayer nbLayer = (BiLstmLayer) nLayer;
				nLayer = nbLayer.ilayer;
			}
			bpttPartialFromNextLayer(nLayer, layer.getRNNNeuroVos(), layer, false, true);
			cLayer.reverseBack();
		}
//		bpttFromNextLayer(layer, false);		
	}
	
	
//	public void updateWw4RNNLayer(ProjectionLayer layer) { 
//		BiProjectionLayer2 currentLayer = (BiProjectionLayer2) layer;
//		super.updateWw4RNNLayer(currentLayer.layer);
//		super.updateWw4RNNLayer(currentLayer.ilayer);
//	}
	

	@Override
	public void bpTT4RNNLayer(LSTMLayer layer) {//assume all the layers are 
		ICell[] allCells = layer.getCells();
		BiLstmLayer currentLayer = (BiLstmLayer) layer;
		
		if (biDirection == NORMAL) {	
			currentLayer.reverseOpposite(sample.length);
			if (layerPos != cfg.layers.length - 1) {//if lstm is the last layer, it means no need bp for it.	
				IRNNLayer nLayer = cfg.layers[layerPos + 1];
				if (nLayer instanceof BiLstmLayer) {
					BiLstmLayer nbLayer = (BiLstmLayer) nLayer;
					nLayer = nbLayer.layer;
				}
				bpttPartialFromNextLayer(nLayer, currentLayer.getRNNNeuroVos(), layer, false, false);
			} else {						
				//
				if (attentionDhj != null) {
					for (int j = 0; j < allCells.length; j++) {
						SimpleNeuroVo vo = allCells[j].getNvTT()[t];
						vo.deltaZz = attentionDhj[t][j];
					}
				} else {
					if (t == tLength - 1) {
						for (int j = 0; j < allCells.length; j++) {
							SimpleNeuroVo vo = allCells[j].getNvTT()[t];
							vo.deltaZz = this.cxtDeltaZz(layerPos)[j];
						}
					} else {
						for (int j = 0; j < allCells.length; j++) {
							SimpleNeuroVo vo = allCells[j].getNvTT()[t];
							vo.deltaZz = 0;
						}
					}
				}
				
				//
			}
			currentLayer.reverseBackOpposite(); 
			
		} else {
			if (layerPos < cfg.layers.length - 2) {
				currentLayer.reverseNormal(sample.length);
				
				IRNNLayer nLayer = cfg.layers[layerPos + 1];
				if (nLayer instanceof BiLstmLayer) {
					BiLstmLayer nbLayer = (BiLstmLayer) nLayer;
					nLayer = nbLayer.ilayer;
				}
				bpttPartialFromNextLayer(nLayer, currentLayer.getRNNNeuroVos(), layer, false, true);				
				
				currentLayer.reverseBackNormal();				
			}		
			/**It is trick here, 
			 * assume the hidden biLstm does not require other externals
			 * **/
			bpTT4PartialRNNLayerCell(currentLayer.layer.getCells(), currentLayer.layer);			 
			bpTT4PartialRNNLayerBlocks(currentLayer.layer.getBlocks(), currentLayer.layer);
			/**It is trick here, 
			 * assume the hidden biLstm does not require other externals
			 * **/
			
			bpTT4PartialRNNLayerCell(currentLayer.ilayer.getCells(), currentLayer.ilayer);				 
			bpTT4PartialRNNLayerBlocks(currentLayer.ilayer.getBlocks(), currentLayer.ilayer);
		}
		
		
//		super.bpTT4RNNLayer(layer);
	}


	@Override
	public void updateWw4RNNLayer(LSTMLayer layer) { 
		BiLstmLayer currentLayer = (BiLstmLayer) layer;
		super.updateWw4RNNLayer(currentLayer.layer);
		super.updateWw4RNNLayer(currentLayer.ilayer);
	}
	
	
}
