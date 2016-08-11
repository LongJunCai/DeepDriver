package deepDriver.dl.aml.cnn;

import java.io.Serializable;

import deepDriver.dl.aml.costFunction.ICostFunction;

public interface ICNNLayer extends Serializable {
	
	public IFeatureMap[] getFeatureMaps();
	
	public double [] featureMaps2Vector();
	
	public void accept(ICNNLayerVisitor visitor);
	
	public ICNNLayer getPreviousLayer();
	
	public ICostFunction getCostFunction();
	
	public LayerConfigurator getLc();

	public void setLc(LayerConfigurator lc);

}
