package deepDriver.dl.aml.linearReg;

public class ParameterScaler {
	double [] maxSet;
	double [] minSet;
	public double [] scaleCoefficients(double [] thetas) {
		for (int i = 0; i < thetas.length; i++) {
			if (maxSet[i] != minSet[i]) {
				thetas[i] = thetas[i]/(maxSet[i] - minSet[i]);
			}
		}
		return thetas;
	}
	public double [] [] scaleParameters(double[][] xVector) {
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
		for (int i = 0; i < xVector.length; i++) {
			double [] x = xVector[i];
			for (int j = 0; j < maxSet.length; j++) {
				if (maxSet[j] != minSet[j]) {
					x[j] = x[j]/(maxSet[j] - minSet[j]);
				}				
			}
		}
		return xVector;
	}
	
	
}
