package deepDriver.dl.aml.lstm;

import java.io.Serializable;
import java.util.Random;

public class RNNNeuroVo implements IRNNNeuroVo, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	static transient Random random = new Random(System.currentTimeMillis());
	double [] wWs;
	double [] rwWs;	
	double [] lwWs;
	
	double [] deltaWWs;
	double [] deltaRwWs;	
	double [] deltaLwWs;
	
	double [] xWWs;
	double [] xRwWs;	
	double [] xLwWs;
	
	transient SimpleNeuroVo [] neuroVos;
	int t;
	boolean inHidenLayer;
	int previousNNN;
//	int nextNNN;
//	public RNNNeuroVo(int t, boolean inHidenLayer, int previousNNN, int currentNNN) {
//		this(t, inHidenLayer, previousNNN, currentNNN, currentNNN);
//	}
	public RNNNeuroVo(){}
	public RNNNeuroVo(int t, boolean inHidenLayer, int previousNNN, int LayerNN, int blockNN, int nextLayerNN) {
		super();
		this.t = t;
		this.inHidenLayer = inHidenLayer;
		this.previousNNN = previousNNN;
		neuroVos = new SimpleNeuroVo[t];
		for (int i = 0; i < neuroVos.length; i++) {
			neuroVos[i] = new SimpleNeuroVo();
		}
		if (previousNNN != 0) {
			wWs = new double[previousNNN + 1];
			this.deltaWWs = new double[previousNNN + 1];
			xWWs = new double[previousNNN + 1];
		} 
		int hiddenNN = 0;
		if (inHidenLayer) {
			if (LayerNN != 0) {
				rwWs = new double[blockNN];
				this.deltaRwWs = new double[blockNN];
				this.xRwWs = new double[blockNN];
				
				lwWs = new double[LayerNN];
				deltaLwWs = new double[LayerNN];
				xLwWs = new double[LayerNN];
			}	
			hiddenNN = LayerNN;
		}
		double b = Math.pow(6.0/(double)(previousNNN + 2 * hiddenNN + nextLayerNN), 0.5);
		length = 2*b;
		min = -b;
		initWeights();
	}
	boolean randomize = true;
	double min = 0;
	double max = 1.0;
	double length = max - min;
	
	private void initWeights() {
		if (randomize) {
			if (wWs != null) {
				for (int i = 0; i < wWs.length; i++) {
					wWs[i] = length * random.nextDouble()
					+ min;
				}
			}
			if (rwWs != null) {
				for (int i = 0; i < rwWs.length; i++) {
					rwWs[i] = length * random.nextDouble()
					+ min;
				}
			}
			
			if (lwWs != null) {
				for (int i = 0; i < lwWs.length; i++) {
					lwWs[i] = length * random.nextDouble()
					+ min;
				}
			}
			
		}		
	}
	
	
	public double[] getLwWs() {
		return lwWs;
	}




	public void setLwWs(double[] lwWs) {
		this.lwWs = lwWs;
	}




	public double[] getDeltaLwWs() {
		return deltaLwWs;
	}




	public void setDeltaLwWs(double[] deltaLwWs) {
		this.deltaLwWs = deltaLwWs;
	}




	public double[] getwWs() {
		return wWs;
	}
	public void setwWs(double[] wWs) {
		this.wWs = wWs;
	}
	public double[] getRwWs() {
		return rwWs;
	}
	public void setRwWs(double[] rwWs) {
		this.rwWs = rwWs;
	}
	public SimpleNeuroVo[] getNvTT() {
		return neuroVos;
	}
	public void setNeuroVos(SimpleNeuroVo[] neuroVos) {
		this.neuroVos = neuroVos;
	}
	public int getT() {
		return t;
	}
	public void setT(int t) {
		this.t = t;
	}
	public boolean isInHidenLayer() {
		return inHidenLayer;
	}
	public void setInHidenLayer(boolean inHidenLayer) {
		this.inHidenLayer = inHidenLayer;
	}
	public int getPreviousNNN() {
		return previousNNN;
	}
	public void setPreviousNNN(int previousNNN) {
		this.previousNNN = previousNNN;
	}
	public double[] getDeltaWWs() {
		return deltaWWs;
	}
	public void setDeltaWWs(double[] deltaWWs) {
		this.deltaWWs = deltaWWs;
	}
	public double[] getDeltaRwWs() {
		return deltaRwWs;
	}
	public void setDeltaRwWs(double[] deltaRwWs) {
		this.deltaRwWs = deltaRwWs;
	}


	public double[] getxWWs() {
		return xWWs;
	}


	public void setxWWs(double[] xWWs) {
		this.xWWs = xWWs;
	}


	public double[] getxRwWs() {
		return xRwWs;
	}


	public void setxRwWs(double[] xRwWs) {
		this.xRwWs = xRwWs;
	}


	public double[] getxLwWs() {
		return xLwWs;
	}


	public void setxLwWs(double[] xLwWs) {
		this.xLwWs = xLwWs;
	}	

}
