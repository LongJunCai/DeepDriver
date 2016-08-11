package deepDriver.dl.aml.ann;

import deepDriver.dl.aml.ann.imp.SparseAutoEncoderLayer;

public class SparseAutoEncoder extends ArtifactNeuroNetworkV2 {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public SparseAutoEncoder() {
		this.aNNCfg = new SparseAutoEncoderCfgFromANNCfg();
	}
	
	public ILayer createLayer() {
		SparseAutoEncoderLayer layer = new SparseAutoEncoderLayer();
		layer.setaNNCfg(aNNCfg);
		if (cf != null) {
			layer.setCostFunction(cf);
		}
		return layer;
	}

}
