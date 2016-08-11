package deepDriver.dl.aml.cnn;

import java.io.Serializable;

public class SubSamplingKernal implements IConvolutionKernal, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	double wW;
	double b;
	
	double deltawW;
	double deltab;
	
	boolean initwW;
	boolean initB;
	
	int ckRows;
	int ckColumns;
	
	int fmapOfPreviousLayer; 
	
	public int getFmapOfPreviousLayer() {
		return fmapOfPreviousLayer;
	}
	public void setFmapOfPreviousLayer(int fmapOfPreviousLayer) {
		this.fmapOfPreviousLayer = fmapOfPreviousLayer;
	}	
	
}
