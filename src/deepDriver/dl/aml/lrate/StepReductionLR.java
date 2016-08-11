package deepDriver.dl.aml.lrate;

import java.io.Serializable;

public class StepReductionLR implements LearningRateManager, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	int stepsCnt = 40000;
	double minLr = 0.0001;
	double reductionRate = 0.1;
	
	int cnt = 0;
	@Override
	public double adjustML(double err, double lrate) {
		cnt ++;
		double nl = lrate * reductionRate;
		if (cnt % stepsCnt == 0 &&  nl >= minLr) {			
			lrate = nl;
		}
		return lrate;
	}
	public int getStepsCnt() {
		return stepsCnt;
	}
	public void setStepsCnt(int stepsCnt) {
		this.stepsCnt = stepsCnt;
	}
	public double getMinLr() {
		return minLr;
	}
	public void setMinLr(double minLr) {
		this.minLr = minLr;
	}
	public double getReductionRate() {
		return reductionRate;
	}
	public void setReductionRate(double reductionRate) {
		this.reductionRate = reductionRate;
	}
	
}
