package deepDriver.dl.aml.ann;

import java.io.Serializable;

public class Normalizer implements Serializable {
	
	private static final long serialVersionUID = 7045724686701473027L;
	
	double [] maxSet;
	double [] minSet;
	
	double maxSet1;
	double minSet1;
	
	public double getScaler() {
		return scaler;
	}
	public void setScaler(double scaler) {
		this.scaler = scaler;
	}
	public double [] scaleCoefficients(double [] thetas) {
		for (int i = 0; i < thetas.length; i++) {
			if (maxSet[i] != minSet[i]) {
				thetas[i] = thetas[i]/(maxSet[i] - minSet[i]);
			}
		}
		return thetas;
	}
	public double [] [] retransformParameters(double[][] xVector) {
		double [] [] output = new double[xVector.length][xVector[0].length];
		for (int i = 0; i < xVector.length; i++) {
			for (int j = 0; j < xVector[0].length; j++) {
				if (maxSet[j] != minSet[j]) {
					output[i][j] = (xVector[i][j]-minSet[j])/(maxSet[j] - minSet[j]);
					/**
					 * prevent the value exceeds the bound.
					 * **/
					if (output[i][j] > 1) {
						output[i][j] = 1;
					}
					if (output[i][j] < 0) {
						output[i][j] = 0;
					}
					/**
					 * prevent the value exceeds the bound.
					 * **/
				}				
			}
		}
		return output;
	}
	
	public double [] [] transformBackParameters(double[][] xVector) {
//		double [] [] output = new double[xVector.length][xVector[0].length];
		double [] [] output = new double[xVector.length][];
		for (int i = 0; i < xVector.length; i++) {
			output[i] = new double[xVector[i].length];
			for (int j = 0; j < xVector[i].length; j++) {
				int mj = j;
				if (j > maxSet.length - 1) {
					mj = maxSet.length - 1;
				}
				output[i][j] = minSet[mj] + (maxSet[mj] - minSet[mj]) * xVector[i][j];
			}
		}
		return output;
	}
	
	private double [][] d122(double [] xVector) {
		double [][] tmp = new double[xVector.length][1];
		for (int i = 0; i < tmp.length; i++) {
			tmp[i][0] = xVector[i];
		}
		return tmp;
	}
	private double [] d221(double [][] xVector) {
		double [] t3 = new double[xVector.length];
		for (int i = 0; i < t3.length; i++) {
			t3[i] = xVector[i][0];
		}
		return t3;
	}
	
	public double [] transformBackParameters(double[] xVector) {
		return d221(transformBackParameters(d122(xVector)));
	}
	
	public double [] transformParameters(double [] xVector) {		
		return d221(transformParameters(d122(xVector)));
	}
	
	double scaler = 1;
	double maxPeak = 0;
	
	public double getMaxPeak() {
		return maxPeak;
	}
	public void setMaxPeak(double maxPeak) {
		this.maxPeak = maxPeak;
	}
	public double [] [] transformParameters(double[][] xVector) {
		maxSet = new double[xVector[0].length];
		minSet = new double[maxSet.length];
		for (int i = 0; i < xVector.length; i++) {
			double [] x = xVector[i];
			for (int j = 0; j < maxSet.length; j++) {
				if (maxSet[j] < x[j]) {
					maxSet[j] = x[j];
				}
				if (minSet[j] > x[j]) {
					minSet[j] = x[j];
				}
			}
		}
		for (int i = 0; i < maxSet.length; i++) {
			if (maxPeak <= 0) {
				maxSet[i] = maxSet[i] * scaler;
			} else {
				maxSet[i] = maxPeak;
			}			
		}
		double [][] changed = new double[xVector.length][xVector[0].length];
		for (int i = 0; i < changed.length; i++) {
//			double [] x = changed[i];
			for (int j = 0; j < maxSet.length; j++) {
				if (maxSet[j] != minSet[j]) {
					if (xVector[i][j] > maxSet[j]) {
						changed[i][j] = 1;
					} else {
						changed[i][j] = (xVector[i][j] - minSet[j])/(maxSet[j] - minSet[j]);
					}					
				}				
			}
		}
		return changed;
	} 
	
	
}
