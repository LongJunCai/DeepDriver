package deepDriver.dl.aml.ann;

import java.io.Serializable;

public class InputParameters implements Serializable {
	
	private static final long serialVersionUID = 9074949404258488114L;
	
//	double alpha = 0.01; //linear
	double alpha = 0.1;
	double [][] input;
	double [] result;
	
	double m = -1;
	
	double [][] result2;
	int iterationNum = 300000;
	
	double lamda = 0.00001;
	
	int [] neuros;
	
	int layerNum = 1;
	
	boolean bpFirstLayer = false;
		
	public boolean isBpFirstLayer() {
		return bpFirstLayer;
	}

	public void setBpFirstLayer(boolean bpFirstLayer) {
		this.bpFirstLayer = bpFirstLayer;
	}

	public double getLamda() {
		return lamda;
	}

	public void setLamda(double lamda) {
		this.lamda = lamda;
	}

	public int[] getNeuros() {
		return neuros;
	}

	public void setNeuros(int[] neuros) {
		this.neuros = neuros;
	}

	public int getLayerNum() {
		if (neuros != null) {
			return neuros.length;
		}
		return layerNum;
	}

	public void setLayerNum(int layerNum) {
		this.layerNum = layerNum;
	}

	public int getIterationNum() {
		return iterationNum;
	}

	public void setIterationNum(int iterationNum) {
		this.iterationNum = iterationNum;
	}

	public double getAlpha() {
		return alpha;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	public double[][] getInput() {
		return input;
	}

	public void setInput(double[][] input) {
		this.input = input;
	}

	public double[] getResult() {
		return result;
	}

	public void setResult(double[] result) {
		this.result = result;
	}

	public double[][] getResult2() {
		return result2;
	}

	public void setResult2(double[][] result2) {
		this.result2 = result2;
	}

	public double getM() {
		return m;
	}

	public void setM(double m) {
		this.m = m;
	}
		

}
