package deepDriver.dl.aml.cnn;

import deepDriver.dl.aml.cnn.cae.IConvAutoEncoderLayerVisitor;

public class SamplingReconstructionLayer extends SamplingLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public SamplingReconstructionLayer(LayerConfigurator lc,
			ICNNLayer previousLayer) {
		super(lc, previousLayer);
	}
	
	public IFeatureMap createIFeatureMap() {
		int r = lc.getCkRows();
		int c = lc.getCkColumns();
		if (lc.getCks() != null) {
			r = lc.getCks()[fmIndex][0];
			c = lc.getCks()[fmIndex][1];
		}
		return new SamplingReconstructionFeatureMap (this,
				lc.getAcf() == null? ActivationFactory.getAf().getTanh(): lc.getAcf()
				, previousLayer, 
				r, c, lc.isFullConnection,
				lc.isFullConnection ? null : lc.getFeatureMapAllocationMatrix()[fmIndex], fmIndex);
	}
	
	public void accept(ICNNLayerVisitor visitor) {
		IConvAutoEncoderLayerVisitor vi = (IConvAutoEncoderLayerVisitor)visitor;
		vi.visitPoolingReconstructionLayer(this);
	}

}
