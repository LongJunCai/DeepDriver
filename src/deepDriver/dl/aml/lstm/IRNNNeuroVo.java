package deepDriver.dl.aml.lstm;

public interface IRNNNeuroVo {
	
	public double[] getLwWs();
	
	public void setLwWs(double[] lwWs);

	public double[] getDeltaLwWs();

	public void setDeltaLwWs(double[] deltaLwWs);
	
	public double[] getwWs() ;
	
	public void setwWs(double[] wWs);
	
	public double[] getRwWs();
	
	public void setRwWs(double[] rwWs);
	
	public SimpleNeuroVo[] getNvTT();
	
	public void setNeuroVos(SimpleNeuroVo[] neuroVos);
	
	public int getT();
	
	public void setT(int t);
	
	public boolean isInHidenLayer();
	
	public void setInHidenLayer(boolean inHidenLayer);
	
	public int getPreviousNNN();
	
	public void setPreviousNNN(int previousNNN);
	
	public double[] getDeltaWWs();
	
	public void setDeltaWWs(double[] deltaWWs);
	
	public double[] getDeltaRwWs();
	
	public void setDeltaRwWs(double[] deltaRwWs);
	
	public double[] getxWWs();


	public void setxWWs(double[] xWWs);


	public double[] getxRwWs();


	public void setxRwWs(double[] xRwWs);


	public double[] getxLwWs();


	public void setxLwWs(double[] xLwWs);

	

}
