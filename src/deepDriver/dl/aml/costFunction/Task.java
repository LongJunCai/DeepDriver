package deepDriver.dl.aml.costFunction;

import java.io.Serializable;

import deepDriver.dl.aml.ann.imp.NeuroUnitImp;

public class Task implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public static int CF_STD = 1;
	public static int CF_SOFTMAX = 2;
	
	int neuroLen;
	int costType;
	int resultLen;
	
	NeuroUnitImp [] nus;
	double [] zZs;
	
	String name;	
	
	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public int getNeuroLen() {
		return neuroLen;
	}
	
	public void setNeuroLen(int neuroLen) {
		this.neuroLen = neuroLen;
	}
	public int getCostType() {
		return costType;
	}
	public void setCostType(int costType) {
		this.costType = costType;
	}

//	public int getResultLen() {
//		return resultLen;
//	}
//
//	public void setResultLen(int resultLen) {
//		this.resultLen = resultLen;
//	}

	public NeuroUnitImp[] getNus() {
		return nus;
	}

	public void setNus(NeuroUnitImp[] nus) {
		this.nus = nus;
	}

	public double[] getzZs() {
		return zZs;
	}

	public void setzZs(double[] zZs) {
		this.zZs = zZs;
	} 
}
