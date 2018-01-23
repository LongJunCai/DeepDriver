package deepDriver.dl.aml.cnn;

import deepDriver.dl.aml.math.MathUtil;

public class BlasCNNWwUpdateVisitor implements ICNNLayerVisitor {

	protected CNNBP bp;
		
	public BlasCNNWwUpdateVisitor(CNNBP bp) {
		super();
		this.bp = bp;
	}

	@Override
	public void visitCNNLayer(CNNLayer layer) {
		IFeatureMap [] fms = layer.getFeatureMaps();
		for (int i = 0; i < fms.length; i++) {
			updateGlobalWws(fms[i]);
		}
		if (layer.getCkM() != null) {
			MathUtil.plus(layer.getCkM(), layer.getDckM(), layer.getCkM());
		}		
	}
	
	private void updateGlobalWws(IFeatureMap fms) {
		fms.setGema(fms.getGema() + fms.getDgamma());
		fms.setBeta(fms.getBeta() + fms.getDbeta());
		if (!bp.useGlobalWeight) {
			return;
		}
		fms.setbB(fms.getbB() + fms.getDeltaBb());
	}

	@Override
	public void visitPoolingLayer(SamplingLayer layer) {
		
	}

	@Override
	public void visitANNLayer(CNNLayer2ANNAdapter layer) {
		
	}

}
