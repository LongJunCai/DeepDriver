package deepDriver.dl.aml.lstm;

import java.io.Serializable;


public class RNNLayer implements IRNNLayer, Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	RNNNeuroVo [] vos0;
	
	LayerCfg lc;
	
	public RNNLayer(int nodeNN, 
			int t, boolean inHidenLayer, int previousNNN, int nextLayerNN, LayerCfg lc) {
		this.lc = lc;
		vos0 = new RNNNeuroVo[nodeNN];
		for (int i = 0; i < vos0.length; i++) {
//			vos0[i] = new RNNNeuroVo(t, inHidenLayer, previousNNN, nodeNN, nodeNN);
			vos0[i] = new RNNNeuroVo(t, inHidenLayer, previousNNN, nodeNN, 0,  nextLayerNN, lc);
		}
	}  

	public LayerCfg getLc() {
		return lc;
	}

	public void setLc(LayerCfg lc) {
		this.lc = lc;
	}

	@Override
	public RNNNeuroVo[] getRNNNeuroVos() { 
		return vos0;
	}

	@Override
	public void fTT(IBPTT bptt) {
		bptt.fTT4RNNLayer(this);
	}
	@Override
	public void bpTT(IBPTT bptt) {
		bptt.bpTT4RNNLayer(this);
	}
	
	public void updateWw(IRNNLayerVisitor bptt) {
		bptt.updateWw4RNNLayer(this);
	}

	@Override
	public void setRNNNeuroVos(RNNNeuroVo[] rnnvos) {
		this.vos0 = rnnvos;
	}

}
