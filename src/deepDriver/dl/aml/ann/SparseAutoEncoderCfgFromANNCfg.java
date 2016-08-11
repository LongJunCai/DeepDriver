package deepDriver.dl.aml.ann;

public class SparseAutoEncoderCfgFromANNCfg extends ANNCfg implements ISparseAutoEncoderCfg {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	double p = 0.05;
	
	double beta = 0.001;

	public double getP() {
		return p;
	}

	public void setP(double p) {
		this.p = p;
	}

	public double getBeta() {
		return beta;
	}

	public void setBeta(double beta) {
		this.beta = beta;
	}	

}
