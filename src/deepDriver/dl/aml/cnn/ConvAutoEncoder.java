package deepDriver.dl.aml.cnn;


public class ConvAutoEncoder {
	
	CNNConfigurator cfg;
	int iterations;
	
	public void construct(LayerConfigurator [] lcs) {
		for (int i = 0; i < lcs.length; i++) {
			LayerConfigurator lc = lcs[i];
			lc.setLast(i == lcs.length - 1);
			if (i == 0) {
				cfg.getLayers()[i] = createCNNLayer(lc, null);
			} else {				
				cfg.getLayers()[i] = createCNNLayer(lc, cfg.getLayers()[i - 1]);
			}		
		}
	}
	
	public ICNNLayer createCNNLayer(LayerConfigurator lc, ICNNLayer previous) {
		ICNNLayer layer = null;
		if (LayerConfigurator.CONVOLUTION_LAYER == lc.getType()) {
			layer = new CNNLayer(lc, previous);
		} else if (LayerConfigurator.POOLING_LAYER == lc.getType()) {
			layer = new SamplingLayer(lc, previous);
		} else if (LayerConfigurator.ANN_LAYER == lc.getType()) {
			layer = new CNNLayer2ANNAdapter(lc, previous);
		} else if (LayerConfigurator.CONV_RECONSTRUCTION_LAYER == lc.getType()) {
			layer = new CNNReconstructionLayer(lc, previous);
		} else if (LayerConfigurator.SAMPLING_RECONSTRUCTION_LAYER == lc.getType()) {
			layer = new SamplingReconstructionLayer(lc, previous);
		}
		return layer;
	}
	
	public void train(IDataStreamPiples idp, int iterations) {
		ConvAeBP convAeBP = new ConvAeBP(this.cfg);
		this.iterations = iterations;
		double error = 0;
		int cnt = 0;
		for (int i = 0; i < iterations; i++) {
			while (idp.hasNext()) {
				IDataMatrix [] ms = idp.next();
				error = error + convAeBP.runTrainEpich(ms, ms[0].getTarget());
				cnt ++;
				if (cnt % 4000 == 0) {
					System.out.println(""+cnt+" samples, the avg error is "+(error/(double)cnt));
				}
			}
		}
	}

}
