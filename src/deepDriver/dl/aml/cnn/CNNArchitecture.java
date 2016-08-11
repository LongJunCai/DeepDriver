package deepDriver.dl.aml.cnn;

import java.util.ArrayList;
import java.util.List;

public class CNNArchitecture {
	
	List<LayerConfigurator> layerCfgs = new ArrayList<LayerConfigurator>();
	
	public List<LayerConfigurator> getLayerCfgs() {
		return layerCfgs;
	}
	
	public void setLayerCfgs(List<LayerConfigurator> layerCfgs) {
		this.layerCfgs = layerCfgs;
	}
	
	public void addLayerCfg(LayerConfigurator lc) {
		layerCfgs.add(lc);
	}
	
}
