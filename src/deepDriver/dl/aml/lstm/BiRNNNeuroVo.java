package deepDriver.dl.aml.lstm;

public class BiRNNNeuroVo extends RNNNeuroVo {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	SimpleNeuroVo [] orignalNvs;
	SimpleNeuroVo [] reverseNvs;
	
	RNNNeuroVo real;
	public BiRNNNeuroVo(RNNNeuroVo vo) {
		this.real = vo;
		this.neuroVos = real.neuroVos;
		this.orignalNvs = real.neuroVos;
		orignalNvs = neuroVos;
		reverseNvs = new SimpleNeuroVo[neuroVos.length];
	}

	public BiRNNNeuroVo(int t, boolean inHidenLayer, int previousNNN,
			int LayerNN, int blockNN, int nextLayerNN, LayerCfg lc) {
		super(t, inHidenLayer, previousNNN, LayerNN, blockNN, nextLayerNN, lc);
		orignalNvs = neuroVos;
		reverseNvs = new SimpleNeuroVo[neuroVos.length];
	}
	
	public void reverse(int lt) {
		int cnt = 0;
		for (int i = lt - 1; i >= 0; i--) {
			reverseNvs[cnt++] = neuroVos[i];
		}
		neuroVos = reverseNvs;
	}
	
	public void reverseBack() {
		neuroVos = orignalNvs;
	}
	
	

}
