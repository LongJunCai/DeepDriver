package deepDriver.dl.aml.cart;

import java.io.Serializable;

public class GbdtParameter implements Serializable {
	Cart cart;
	double r;
	double [] currentTrainingInVars;
	double [] currentTestInVars;
	public Cart getCart() {
		return cart;
	}
	public void setCart(Cart cart) {
		this.cart = cart;
	}
	public double getR() {
		return r;
	}
	public void setR(double r) {
		this.r = r;
	}
	public double[] getCurrentTrainingInVars() {
		return currentTrainingInVars;
	}
	public void setCurrentTrainingInVars(double[] currentTrainingInVars) {
		this.currentTrainingInVars = currentTrainingInVars;
	}
	public double[] getCurrentTestInVars() {
		return currentTestInVars;
	}
	public void setCurrentTestInVars(double[] currentTestInVars) {
		this.currentTestInVars = currentTestInVars;
	}

}
