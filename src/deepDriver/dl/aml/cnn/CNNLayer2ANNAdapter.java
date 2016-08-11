package deepDriver.dl.aml.cnn;

import java.io.Serializable;


import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.ann.imp.LayerImpV2;
import deepDriver.dl.aml.ann.imp.NeuroUnitImp;
import deepDriver.dl.aml.costFunction.ICostFunction;
import deepDriver.dl.aml.costFunction.SoftMax4ANN;
import deepDriver.dl.aml.random.RandomFactory;

public class CNNLayer2ANNAdapter implements ICNNLayer, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	LayerImpV2 layerImpV2;
	LayerConfigurator lc;
	ICNNLayer previousLayer;
	IActivationFunction acf = ActivationFactory.getAf().getAcf();
	static {
		NeuroUnitImp.random = RandomFactory.getRandom();
	}
	ICostFunction costFunction;

	@Override
	public ICostFunction getCostFunction() {
		return costFunction;
	}	
	public CNNLayer2ANNAdapter(LayerConfigurator lc, ICNNLayer previousLayer) {
		this.lc = lc;
		this.previousLayer = previousLayer;
		layerImpV2 = new LayerImpV2();
		//add ann cfg support.
		layerImpV2.setaNNCfg(lc.aNNCfg);
		//
		if (lc.isLast()) {
			if (lc.getCostFunction() != null) {
				costFunction = lc.getCostFunction();
			} else {
				costFunction = new SoftMax4ANN();//by default it is softmax used.
			}
			layerImpV2.setCostFunction(costFunction);
		}
		if (previousLayer instanceof CNNLayer2ANNAdapter) {
			CNNLayer2ANNAdapter pl = (CNNLayer2ANNAdapter) previousLayer;
			layerImpV2.buildup(pl.layerImpV2, null, 
					lc.getAcf() == null? ActivationFactory.getAf().getAcf(): lc.getAcf(), 
							lc.isLast(), lc.getFeatureMapNum());
		} else {			
			layerImpV2.buildup(null, new double [][] {
					previousLayer.featureMaps2Vector()}, 
					lc.getAcf() == null? ActivationFactory.getAf().getAcf(): lc.getAcf(),
							lc.isLast(), lc.getFeatureMapNum());

		}
	}
	
	public LayerConfigurator getLc() {
		return lc;
	}

	public void setLc(LayerConfigurator lc) {
		this.lc = lc;
	}
	
	@Override
	public IFeatureMap[] getFeatureMaps() {
		return null;
	}
	@Override
	public double[] featureMaps2Vector() {
		return null;
	}

	@Override
	public void accept(ICNNLayerVisitor visitor) {
		visitor.visitANNLayer(this);
	}

	@Override
	public ICNNLayer getPreviousLayer() {
		return previousLayer;
	}
	
	

}
