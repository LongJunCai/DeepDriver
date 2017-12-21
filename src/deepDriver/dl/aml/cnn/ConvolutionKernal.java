package deepDriver.dl.aml.cnn;

import java.io.Serializable;

public class ConvolutionKernal implements IConvolutionKernal, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	int fmapOfPreviousLayer;
	double [][] wWs;
	double b;
	
	double [][] detalwWs;
	double deltab;
	boolean [][] initDeltaZzs;
	boolean initB;
	
	
	public int getFmapOfPreviousLayer() {
		return fmapOfPreviousLayer;
	}
	public void setFmapOfPreviousLayer(int fmapOfPreviousLayer) {
		this.fmapOfPreviousLayer = fmapOfPreviousLayer;
	}
	public boolean[][] getInitDeltaZzs() {
		return initDeltaZzs;
	}
	public void setInitDeltaZzs(boolean[][] initDeltaZzs) {
		this.initDeltaZzs = initDeltaZzs;
	}
	public boolean isInitB() {
		return initB;
	}
	public void setInitB(boolean initB) {
		this.initB = initB;
	}	
	
}
