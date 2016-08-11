package deepDriver.dl.aml.fnn;

import deepDriver.dl.aml.cnn.ConvolutionNeuroNetwork;
import deepDriver.dl.aml.cnn.FractalBlock;
import deepDriver.dl.aml.cnn.ICNNLayer;
import deepDriver.dl.aml.cnn.LayerConfigurator;

public class FractalNet extends ConvolutionNeuroNetwork {	
	
	public ICNNLayer createCNNLayer(LayerConfigurator lc, ICNNLayer previous) {
		ICNNLayer layer = null;
		if (LayerConfigurator.FRACTAL_BLOCK_LAYER == lc.getType()) {
			layer = new FractalBlock(lc, previous);
			return layer;
		} 
		return super.createCNNLayer(lc, previous);
	}

}
