package deepDriver.dl.aml.costFunction;

import java.io.Serializable;

public class PositiveTask extends Task implements Serializable {
	private static final long serialVersionUID = 1L;
	
	public boolean checkRule(double [] target) {
		for (int i = 0; i < target.length; i++) {
			if (target[i] < 0) {
				return false;
			}
		}
		return true;
	}
}
