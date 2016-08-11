package deepDriver.dl.aml.cnn;

import java.io.Serializable;

public class FeatureMapTag implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	int r;
	int c;
	public int getR() {
		return r;
	}
	public void setR(int r) {
		this.r = r;
	}
	public int getC() {
		return c;
	}
	public void setC(int c) {
		this.c = c;
	}

}
