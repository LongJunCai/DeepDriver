package deepDriver.dl.aml.lstm;

import java.io.Serializable;

import deepDriver.dl.aml.lstm.imp.Block;

public class BiLstmLayer extends LSTMLayer implements IRNNLayer, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	LSTMLayerV2 layer;
	LSTMLayerV2 ilayer;
	
	BiCell [] cells;
	Block [] blocks;
	public BiLstmLayer(int nodeNN, int t, boolean inHidenLayer, int previousNNN, int nextLayerNN, LayerCfg lc) {
//		super(nodeNN, t, inHidenLayer, previousNNN, nextLayerNN);
		//include 2 LSTM Layers.
		layer = new LSTMLayerV2(nodeNN, t, inHidenLayer, previousNNN, nextLayerNN, lc);
		ilayer = new LSTMLayerV2(nodeNN, t, inHidenLayer, previousNNN, nextLayerNN, lc);
		
		cells = new BiCell[nodeNN * 2];
		blocks = new Block[layer.getBlocks().length + ilayer.getBlocks().length];
		int cnt = 0;
		int bcnt = 0;
		for (int i = 0; i < layer.getRNNNeuroVos().length; i++) {
			cells[cnt ++] = new BiCell(layer.getRNNNeuroVos()[i]);			
		}
		for (int i = 0; i < ilayer.getRNNNeuroVos().length; i++) {
			cells[cnt ++] = new BiCell(ilayer.getRNNNeuroVos()[i]);
		}
		
		for (int i = 0; i < layer.getBlocks().length; i++) {
			blocks[bcnt++] = layer.getBlocks()[i];
		}
		for (int i = 0; i < ilayer.getBlocks().length; i++) {
			blocks[bcnt++] = ilayer.getBlocks()[i];
		}
	}
	
	int Normal = 1;
	int biN = 2;
	
	public void reverseNormal(int lt) {
		reverse(Normal, lt);
	}
	
	public void reverseOpposite(int lt) {
		reverse(biN, lt);
	}
	
	public void reverseBackNormal() {
		reverseBack(Normal);
	}
	public void reverseBackOpposite() {
		reverseBack(biN);
	}
	
	public void reverse(int state, int lt) {
		if (state == Normal) {
			for (int i = 0; i < layer.getRNNNeuroVos().length; i++) {
				cells[i].reverse(lt);
			}
		} else {
			for (int i = 0; i < ilayer.getRNNNeuroVos().length; i++) {
				cells[layer.getRNNNeuroVos().length + i].reverse(lt);
			}
		}
	}
	
	public void reverseBack(int state) {
		if (state == Normal) {
			for (int i = 0; i < layer.getRNNNeuroVos().length; i++) {
				cells[i].reverseBack();
			}
		} else {
			for (int i = 0; i < ilayer.getRNNNeuroVos().length; i++) {
				cells[layer.getRNNNeuroVos().length + i].reverseBack();
			}
		}
	}

	@Override
	public void fTT(IBPTT bptt) {
		bptt.fTT4RNNLayer(this);
	}

	@Override
	public void bpTT(IBPTT bptt) {
		bptt.bpTT4RNNLayer(this);
	}

	@Override
	public RNNNeuroVo[] getRNNNeuroVos() {
		return cells;
	}

	@Override
	public void setRNNNeuroVos(RNNNeuroVo[] rnnvos) {
	}

	@Override
	public void updateWw(IRNNLayerVisitor visitor) {
		visitor.updateWw4RNNLayer(this);
	}

	@Override
	public Block[] getBlocks() {
		return blocks;
	}

	@Override
	public void setBlocks(Block[] blocks) {
		this.blocks = blocks;
	}

	@Override
	public ICell[] getCells() {
		return cells;
	}

}
