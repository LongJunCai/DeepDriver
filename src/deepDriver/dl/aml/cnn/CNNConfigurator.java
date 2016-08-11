package deepDriver.dl.aml.cnn;

import java.io.Serializable;

public class CNNConfigurator implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	double l = 0.001; 
	double m = 0.8;	
	  
	double acc = 0.00001;
	
	String name = "";
	int threadsNum = 1;
	
	ICNNLayer [] layers;
	
	int kLength;
	boolean withFC;
	
	public static int AVG_POOLING_TYPE = 1;
	public static int MAX_POOLING_TYPE = 2;
	int poolingType = AVG_POOLING_TYPE;	
		
	public boolean isWithFC() {
		return withFC;
	}

	public void setWithFC(boolean withFC) {
		this.withFC = withFC;
	}

	public int getkLength() {
		return kLength;
	}

	public void setkLength(int kLength) {
		this.kLength = kLength;
	}

	public int getThreadsNum() {
		return threadsNum;
	}

	public void setThreadsNum(int threadsNum) {
		this.threadsNum = threadsNum;
	}

	public int getPoolingType() {
		return poolingType;
	}

	public void setPoolingType(int poolingType) {
		this.poolingType = poolingType;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}
	
	boolean useBN = false;
	

	public boolean isUseBN() {
		return useBN;
	}

	public void setUseBN(boolean useBN) {
		this.useBN = useBN;
	}

	public double getL() {
		return l;
	}

	public void setL(double l) {
		this.l = l;
	}

	public double getM() {
		return m;
	}

	public void setM(double m) {
		this.m = m;
	}

	public ICNNLayer[] getLayers() {
		return layers;
	}

	public void setLayers(ICNNLayer[] layers) {
		this.layers = layers;
	}	

}
