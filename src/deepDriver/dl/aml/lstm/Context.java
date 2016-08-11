package deepDriver.dl.aml.lstm;

import java.io.Serializable;

public class Context implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	ContextLayer [] contextLayers;

	public ContextLayer[] getContextLayers() {
		return contextLayers;
	}

	public void setContextLayers(ContextLayer[] contextLayers) {
		this.contextLayers = contextLayers;
	}	
	
	public Context() {
		
	}
	
	double [] preCxtSc = null;	
	double [] preCxtAa = null;
	public Context(double[] preCxtSc, double[] preCxtAa) {
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
