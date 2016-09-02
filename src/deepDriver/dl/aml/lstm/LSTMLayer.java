package deepDriver.dl.aml.lstm;

import java.io.Serializable;

import deepDriver.dl.aml.lstm.imp.Block;

public class LSTMLayer implements IRNNLayer, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	Block [] blocks;
	LayerCfg lc;
	public LSTMLayer() {
		
	}
	
	public LSTMLayer(int nodeNN, int t, boolean inHidenLayer, int previousNNN, int nextLayerNN, LayerCfg lc) {
		blocks = new Block[1];
		this.lc = lc;
		blocks[0] = new Block(nodeNN, nodeNN, t, inHidenLayer, previousNNN, nextLayerNN, lc);
		
	}
	
	public LayerCfg getLc() {
		return lc;
	}

	public void setLc(LayerCfg lc) {
		this.lc = lc;
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
