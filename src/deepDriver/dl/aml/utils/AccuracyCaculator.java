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
	
	public boolean check(double [] ta, double [] tb) {
		if (getMaxPos(ta) == getMaxPos(tb)) {
			return true;
		}
		return false;
	}
	
	public int getMaxPos(double [] ta) {
		int pos = 0;
		for (int i = 0; i < ta.length; i++) {
			if (ta[i] > ta[pos]) {
				pos = i;
			}
		}
		return pos;
	}
	
	int cnt = 0;
	int correctCnt = 0;
	
	public void cntIncrease() {
		cnt ++;
	}
	
	public void correctCntIncrease() {
		correctCnt ++;
	}
	
	public void reset() {
		cnt = 0;
		correctCnt = 0;
	}
	
	int summaryInterval = 200;
	
	public void summaryCp() {
		if (cnt % summaryInterval == 0) {
			summary();
		}
	}
	
	public void summary() {
		System.out.println("All count is: "+ cnt
				+", the correct count is: "+correctCnt
				+", the accuracy is: "+ (double)correctCnt/(double)cnt);
	}

}
