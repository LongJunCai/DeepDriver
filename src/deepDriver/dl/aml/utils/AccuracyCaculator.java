package deepDriver.dl.aml.utils;

public class AccuracyCaculator {	
	public double caculateAccuracy(double [] x, double [] px) {
		double avgX = 0;
		double sumX = 0;
		double stdVar = 0;
		double err = 0;
		for (int i = 0; i < x.length; i++) {
			sumX = sumX + x[i];
		}
		avgX = sumX/(double)x.length;
		for (int i = 0; i < x.length; i++) {
			stdVar = stdVar + (x[i] - avgX) *  (x[i] - avgX);
			err = err+ (x[i] - px[i]) *  (x[i] - px[i]);
		}
		return 1 - err/stdVar;
	}

}
