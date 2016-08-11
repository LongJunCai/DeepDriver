package deepDriver.dl.aml.cnn;

public interface ICNNLayerVisitor {
	
	public void visitCNNLayer(CNNLayer layer);
	
	public void visitPoolingLayer(SamplingLayer layer);
	
	public void visitANNLayer(CNNLayer2ANNAdapter layer);

}
