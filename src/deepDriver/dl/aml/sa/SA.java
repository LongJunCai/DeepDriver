package deepDriver.dl.aml.sa;

import java.util.Random;

public class SA {
	
	double defaultT = 500;
	double t = defaultT;
	double r = 0.5;
	
	boolean optimMax = true;
	
	int initLoop = 1;
	
	Random rd = new Random(System.currentTimeMillis());
		
	public SA(double defaultT, double r, boolean optimMax, int initLoop) {
		super();
		this.r = r;
		this.defaultT = defaultT;
		t = defaultT;
		this.optimMax = optimMax;
		this.initLoop = initLoop;
		if (initLoop > 0 ) {
			for (int i = 0; i < initLoop; i++) {
				t = r * t;
			}
		}		
	}
	
	public void reset() {
		t = defaultT;
	}
	
	public boolean sa(double deltaE) {
		if (!optimMax) {
			deltaE = -1.0 * deltaE;
		}
		t = t * r;
		double d = rd.nextDouble();
		if (deltaE > 0) {
			return true;
		}
		if (Math.exp(deltaE / t)  > d) {
			return true;
		}		
		return false;
	}

}
