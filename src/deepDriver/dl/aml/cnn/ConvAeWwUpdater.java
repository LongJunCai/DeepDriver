package deepDriver.dl.aml.cnn;

import deepDriver.dl.aml.cnn.cae.IConvAutoEncoderLayerVisitor;

public class ConvAeWwUpdater extends CNNWwUpdateVisitor implements
		IConvAutoEncoderLayerVisitor {

	@Override
	public void visitCNNReconstructionLayer(CNNReconstructionLayer layer) {
		visitCNNLayer(layer);
	}

	@Override
	public void visitPoolingReconstructionLayer(
			SamplingReconstructionLayer layer) {
		visitPoolingLayer(layer);
	}

}
