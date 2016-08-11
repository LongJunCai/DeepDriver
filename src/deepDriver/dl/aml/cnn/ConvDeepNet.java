package deepDriver.dl.aml.cnn;

import java.util.List;


public class ConvDeepNet extends ConvolutionNeuroNetwork {
	
	CNNArchitecture architecture;
	public void construct(CNNArchitecture architecture, CNNConfigurator cfg) {
		this.architecture = architecture;
		this.cfg = cfg;						
	}
	
	public void train(IDataStream is, IDataStream tis) throws Exception {
		learnSelf(is);
		tuneFine(is);
	}
	int iterations = 10;
	int index = 0;
	public void learnSelf(IDataStream is) {
		List<LayerConfigurator>  cfgs = architecture.getLayerCfgs();
		cfg.layers = new ICNNLayer[cfgs.size()];
		
		for (int i = 0; i < cfgs.size(); i++) {
			index = i;
			LayerConfigurator lc = cfgs.get(i);
			lc.setLast(i == cfgs.size() - 1);
			if (i == 0) {
				cfg.layers[i] = createCNNLayer(lc, null);
			} else {
//				cfg.layers[i] = createCNNLayer(lc, cfg.layers[i - 1]);
				ConvAutoEncoder cae = new ConvAutoEncoder();
				cae.construct(new LayerConfigurator[] {
						replicate(cfgs.get(i - 1), false),
						replicate(cfgs.get(i), false), 
						replicate(cfgs.get(i - 1), true)});
				cae.train(new DataStreamPiples(), iterations);
			}
			
		}
	}
	
	public void forwardCNN() {
		
	}
	
	class DataStreamPiples implements IDataStreamPiples {

		@Override
		public IDataMatrix[] next() {
			forwardCNN();
			
//			IDataMatrix[] 
			return null;
		}

		@Override
		public boolean hasNext() {
			return false;
		}

		@Override
		public boolean reset() {
			return false;
		}
		
	}
	
	public LayerConfigurator replicate(LayerConfigurator lc, boolean reconstructed) {
		return null;
	}
	
	
	public void tuneFine(IDataStream is) {
		
	}
	
	
	
	public void test(IDataStream tis) {
		
	}
	

}
