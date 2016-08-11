package deepDriver.dl.aml.ann;

import java.io.Serializable;

public class ANNCfg implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	double dropOut = 0;
	
	boolean isTesting;
	
	int threadsNum = 1;	

	public int getThreadsNum() {
		return threadsNum;
	}

	public void setThreadsNum(int threadsNum) {
		this.threadsNum = threadsNum;
	}

	public boolean isTesting() {
		return isTesting;
	}

	public void setTesting(boolean isTesting) {
		this.isTesting = isTesting;
	}

	public double getDropOut() {
		return dropOut;
	}

	public void setDropOut(double dropOut) {
		this.dropOut = dropOut;
	}
	
	
}
