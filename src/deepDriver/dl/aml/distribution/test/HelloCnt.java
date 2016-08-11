package deepDriver.dl.aml.distribution.test;

import java.io.Serializable;

public class HelloCnt implements Serializable {
	/**
	 * 
	 */
	transient int [] k = new int[30];
	private int i = 0;
	private int j = 0;	
	
	public int[] getK() {
		return k;
	}
	public void setK(int[] k) {
		this.k = k;
	}
	public int getI() {
		return i;
	}
	public void setI(int i) {
		this.i = i;
	}
	public int getJ() {
		return j;
	}
	public void setJ(int j) {
		this.j = j;
	}
	
}
