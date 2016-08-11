package deepDriver.dl.aml.lstm;

import java.io.Serializable;

public class SimpleNeuroVo implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	boolean dropOut = false;
	double aA;
	double zZ;
	double deltaZz;
	public double getaA() {
		return aA;
	}
	public void setaA(double aA) {
		this.aA = aA;
	}
	public double getzZ() {
		return zZ;
	}
	public void setzZ(double zZ) {
		this.zZ = zZ;
	}
	public double getDeltaZz() {
		return deltaZz;
	}
	public void setDeltaZz(double deltaZz) {
		this.deltaZz = deltaZz;
	}
	public boolean isDropOut() {
		return dropOut;
	}
	public void setDropOut(boolean dropOut) {
		this.dropOut = dropOut;
	}		
}
