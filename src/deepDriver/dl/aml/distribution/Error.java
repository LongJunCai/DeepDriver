package deepDriver.dl.aml.distribution;

import java.io.Serializable;

public class Error implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	double err;
	int cnt;
	boolean ready;
	
	public boolean isReady() {
		return ready;
	}
	public void setReady(boolean ready) {
		this.ready = ready;
	}
	public double getErr() {
		return err;
	}
	public void setErr(double err) {
		this.err = err;
	}
	public int getCnt() {
		return cnt;
	}
	public void setCnt(int cnt) {
		this.cnt = cnt;
	}
	
	

}
