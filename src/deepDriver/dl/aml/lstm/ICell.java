package deepDriver.dl.aml.lstm;

public interface ICell extends IRNNNeuroVo {
	
	public double [] getSc();

	public void setSc(double [] sc);
	
	public double[] getDeltaSc();

	public void setDeltaSc(double[] deltaSc);
	
	public double[] getCZz();

	public void setCZz(double[] scZz);
	
	public double[] getDeltaC();

	public void setDeltaC(double[] deltaC);

}
