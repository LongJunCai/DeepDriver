package deepDriver.dl.aml.cnn;

import java.io.Serializable;

public class SamplingLayer extends CNNLayer implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public SamplingLayer(LayerConfigurator lc, ICNNLayer previousLayer) {
		super(lc, previousLayer);
	}
	
	public IFeatureMap createIFeatureMap() {
		int r = lc.getCkRows();
		int c = lc.getCkColumns();
		if (lc.getCks() != null) {
			r = lc.getCks()[fmIndex][0];
			c = lc.getCks()[fmIndex][1];
		}
		return new SamplingFeatureMap(this,
				lc.getAcf() == null? ActivationFactory.getAf().getTanh(): lc.getAcf()
//				lc.getAcf() == null? ActivationFactory.getAf().getReLU(): lc.getAcf()

				, previousLayer, 
				r, c, lc.isFullConnection,
				lc.isFullConnection ? null : lc.getFeatureMapAllocationMatrix()[fmIndex], fmIndex);
	}
	
	@Override
	public void accept(ICNNLayerVisitor visitor) {
		visitor.visitPoolingLayer(this);
	}

}
