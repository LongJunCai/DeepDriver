package deepDriver.dl.aml.lstm;

import java.io.Serializable;

import deepDriver.dl.aml.lstm.imp.Block;
import deepDriver.dl.aml.lstm.imp.Cell;

public class LSTMLayerV2 extends LSTMLayer implements IRNNLayer, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	Block [] blocks;
	Cell [] cells;
	int cellN = 1;
	public LSTMLayerV2(int nodeNN, int t, boolean inHidenLayer, int previousNNN, int nextLayerNN, LayerCfg lc) {
		super(nodeNN, t, inHidenLayer, previousNNN, nextLayerNN, lc);
		blocks = new Block[nodeNN];
		cells = new Cell[nodeNN * cellN];
		int cnt = 0;
		for (int i = 0; i < blocks.length; i++) {
			blocks[i] = new Block(cells.length, cellN, t, inHidenLayer, previousNNN, nextLayerNN, lc);
			Cell [] bcs = (Cell[]) blocks[i].getCells();
			for (int j = 0; j < bcs.length; j++) {
				cells[cnt ++] = bcs[j];
			}
		}				
	}
	@Override
	public RNNNeuroVo[] getRNNNeuroVos() { 
		return cells;
	}
	
	public ICell [] getCells() { 
		return cells;
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
	
}
