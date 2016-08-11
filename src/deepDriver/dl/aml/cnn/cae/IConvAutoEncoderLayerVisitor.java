package deepDriver.dl.aml.cnn.cae;

import deepDriver.dl.aml.cnn.CNNReconstructionLayer;
import deepDriver.dl.aml.cnn.ICNNLayerVisitor;
import deepDriver.dl.aml.cnn.SamplingReconstructionLayer;

public interface IConvAutoEncoderLayerVisitor extends ICNNLayerVisitor {

	public void visitCNNReconstructionLayer(CNNReconstructionLayer layer);
	
	public void visitPoolingReconstructionLayer(SamplingReconstructionLayer layer);
}
