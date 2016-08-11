package deepDriver.dl.aml.lstm;

import java.io.Serializable;

public class ContextLayer implements Serializable {
	
	private static final long serialVersionUID = 1L;
	double [] preCxtSc = null;	
	double [] preCxtAa = null;
	public ContextLayer(double[] preCxtSc, double[] preCxtAa) {
		super();
		this.preCxtSc = preCxtSc;
		this.preCxtAa = preCxtAa;
	}
	public double[] getPreCxtSc() {
		return preCxtSc;
	}
	public void setPreCxtSc(double[] preCxtSc) {
		this.preCxtSc = preCxtSc;
	}
	public double[] getPreCxtAa() {
		return preCxtAa;
	}
	public void setPreCxtAa(double[] preCxtAa) {
		this.preCxtAa = preCxtAa;
	}	

}
