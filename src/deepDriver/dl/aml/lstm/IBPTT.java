package deepDriver.dl.aml.lstm;


public interface IBPTT extends IRNNLayerVisitor {
	
	public Context getPreCxts();

	public void setPreCxts(Context preCxts);
	
	public Context getHLContext();
	
	public double [][] fTT(double [][] sample, boolean test);
	
	public double runEpich(double [][] sample, 
			double [][] targets);
	
	public void fTT4RNNLayer(RNNLayer layer);
	
	public void fTT4RNNLayer(LSTMLayer layer);
	
	public void bpTT4RNNLayer(RNNLayer layer);
	
	public void bpTT4RNNLayer(LSTMLayer layer);
	
	public void fTT4RNNLayer(ProjectionLayer layer);
	
	public void bpTT4RNNLayer(ProjectionLayer layer);
	
//	public void fTT4RNNLayer(BiLstmLayer layer);
//	
//	public void bpTT4RNNLayer(BiLstmLayer layer);
	
//	public void updateWw4RNNLayer(RNNLayer layer);
//	
//	public void updateWw4RNNLayer(LSTMLayer layer);

}
