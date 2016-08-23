package deepDriver.dl.aml.lstm;

public interface IRNNLayerVisitor {
	
	public void updateWw4RNNLayer(RNNLayer layer);
	
	public void updateWw4RNNLayer(LSTMLayer layer);

	public void updateWw4RNNLayer(ProjectionLayer layer);	
	
//	public void updateWw4RNNLayer(BiLstmLayer layer);
	
}
