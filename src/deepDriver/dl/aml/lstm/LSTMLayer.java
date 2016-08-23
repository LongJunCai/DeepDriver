package deepDriver.dl.aml.lstm;

import java.io.Serializable;

import deepDriver.dl.aml.lstm.imp.Block;

public class LSTMLayer implements IRNNLayer, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	Block [] blocks;
	public LSTMLayer() {
		
	}
	
	public LSTMLayer(int nodeNN, int t, boolean inHidenLayer, int previousNNN, int nextLayerNN) {
		blocks = new Block[1];
		blocks[0] = new Block(nodeNN, nodeNN, t, inHidenLayer, previousNNN, nextLayerNN);
		
	}
	@Override
	public RNNNeuroVo[] getRNNNeuroVos() { 
		return blocks[0].getRNNNeuroVos();
	}
	@Override
	public void fTT(IBPTT bptt) {
		bptt.fTT4RNNLayer(this);
	}
	@Override
	public void bpTT(IBPTT bptt) {
		bptt.bpTT4RNNLayer(this);
	}
	public Block[] getBlocks() {
		return blocks;
	}
	public void setBlocks(Block[] blocks) {
		this.blocks = blocks;
	}
	@Override
	public void updateWw(IRNNLayerVisitor bptt) {
		bptt.updateWw4RNNLayer(this);
	}
	@Override
	public void setRNNNeuroVos(RNNNeuroVo[] rnnvos) {
	}

	public ICell[] getCells() {
		return blocks[0].getCells();
	}
	
}
