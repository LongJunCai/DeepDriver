package deepDriver.dl.aml.lstm.imp;

import java.io.Serializable;

import deepDriver.dl.aml.lstm.ICell;
import deepDriver.dl.aml.lstm.LayerCfg;
import deepDriver.dl.aml.lstm.RNNNeuroVo;

public class Cell extends RNNNeuroVo implements ICell, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	transient double [] sc;	
	transient double [] deltaSc;
	
	transient double [] cZz;
	transient double [] deltaC;	

	public Cell(int t, boolean inHidenLayer, int previousNNN, int layerNN, int blockNN, int nextLayerNN,
			LayerCfg lc) {
		super(t, inHidenLayer, previousNNN, layerNN, blockNN, nextLayerNN, lc);
	}

	public double [] getSc() {
		return sc;
	}

	public void setSc(double [] sc) {
		this.sc = sc;
	}

	public double[] getDeltaSc() {
		return deltaSc;
	}

	public void setDeltaSc(double[] deltaSc) {
		this.deltaSc = deltaSc;
	}

	public double[] getCZz() {
		return cZz;
	}

	public void setCZz(double[] scZz) {
		this.cZz = scZz;
	}

	public double[] getDeltaC() {
		return deltaC;
	}

	public void setDeltaC(double[] deltaC) {
		this.deltaC = deltaC;
	}
	
	
	
}
