package deepDriver.dl.aml.dnn;


import deepDriver.dl.aml.ann.ArtifactNeuroNetworkV2;
import deepDriver.dl.aml.ann.ILayer;
import deepDriver.dl.aml.ann.InputParameters;
import deepDriver.dl.aml.ann.SparseAutoEncoder;
import deepDriver.dl.aml.ann.imp.LayerImpV2;

public class DNN extends ArtifactNeuroNetworkV2 {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public void trainModel(InputParameters parameters) {
		double d = getaNNCfg().getDropOut();
		getaNNCfg().setDropOut(0);
		learnSelf(parameters);
		getaNNCfg().setDropOut(d);
		tuneFine(parameters);
	}
//	Normalizer normalizer = new Normalizer();
	ILayer firstLayer;
	
	int slLoopNum = 50;
	double slLamda = 0;
	
	public void learnSelf(InputParameters orignalPramameters) {
		double [][] input = normalizer.transformParameters(orignalPramameters.getInput());
		int ln = orignalPramameters.getLayerNum();
		firstLayer = createLayer();
		slLamda = orignalPramameters.getLamda();
		this.setFirstLayer(firstLayer);
		debugPrint("Begin to build up the DNN, start to pre-training first");
		firstLayer.buildup(null, input, createAcf(), false, input[0].length);
//		ILayer currentLayer = firstLayer;
		boolean sl4LastLayer = true;
		if (sl4LastLayer) {
		} else {
			ln = ln - 1;
		}
		for (int i = 0; i < ln; i++) {
			int hiddenNc = input[0].length;
			if (orignalPramameters.getNeuros() != null) {
				hiddenNc = orignalPramameters.getNeuros()[i];
			}
			System.out.println("Pre-training for layer "+(i + 1));
			double [][] newInput = caculateHiddenInputs(input);
			ILayer last= getLastLayer();
			InputParameters newPramaters = new InputParameters();
			newPramaters.setInput(newInput);
			newPramaters.setAlpha(orignalPramameters.getAlpha());
			newPramaters.setLamda(slLamda);
			
			newPramaters.setIterationNum(slLoopNum);
			int nc = newInput[0].length;
			
//			SparseAutoEncoderCfgFromANNCfg annCfg4DNN = new SparseAutoEncoderCfgFromANNCfg();
//			annCfg4DNN.setP(0.05);
			SparseAutoEncoder sae = new SparseAutoEncoder();
			sae.setUseNormalizer(false);
			if (i == ln - 1) {
				System.out.println("Pre-training for last layer. "+(i + 1));
				newPramaters.setResult2(orignalPramameters.getResult2());
				newPramaters.setResult(orignalPramameters.getResult());
				newPramaters.setNeuros(new int[] {hiddenNc});					
				sae.setCf(getCf());
				sae.setkLength(getkLength());
			} else {
				newPramaters.setResult2(newInput);
				newPramaters.setNeuros(new int[] {hiddenNc, nc});	
			}				
						
			sae.trainModel(newPramaters);
			System.out.println("Done for layer "+(i + 1)+" learning.");
			ILayer newHiddenLayer = sae.getFirstLayer().getNextLayer();		
			newHiddenLayer.setPos(i+1);
			last.setNextLayer(newHiddenLayer);
			newHiddenLayer.setNextLayer(null);
			newHiddenLayer.setPreviousLayer(last);			
			LayerImpV2 l2 = (LayerImpV2) newHiddenLayer;
			l2.setaNNCfg(getaNNCfg());
		}
		if (!sl4LastLayer) {
			ILayer last= getLastLayer();
			ILayer newLayer = createLayer();
			newLayer.setPos(orignalPramameters.getLayerNum());
			newLayer.buildup(last, input, createAcf(), 
				true, orignalPramameters.getNeuros()[orignalPramameters.getNeuros().length - 1]);
		}

	}
	
	
	public double [][] caculateHiddenInputs(double [][] orginalInput) {		
		double [][] newInput = new double[orginalInput.length][];
		for (int i = 0; i < newInput.length; i++) {
			double [][] x = new double[][]{orginalInput[i]};
			ILayer cl = firstLayer;
			ILayer lastLayer = firstLayer;
			while (cl != null) {
				cl.forwardPropagation(x);
				lastLayer = cl;
				cl = cl.getNextLayer();		
			}
			newInput[i] = new double[lastLayer.getNeuros().size()];
			for (int j = 0; j < newInput[i].length; j++) {
				newInput[i][j] = lastLayer.getNeuros().get(j).getAaz(0);
			}
		}
		return newInput;
	}
	
	public ILayer getLastLayer() {
		ILayer cl = firstLayer;
		ILayer lastLayer = firstLayer;
		while (cl != null) {
			lastLayer = cl;
			cl = cl.getNextLayer();		
		}
		return lastLayer;
	}

	double acc = 0.1;
	public void tuneFine(InputParameters parameters) {
		System.out.println("Prepare for fine tuning.");
		double [][] input = 
				normalizer.transformParameters(parameters.getInput());
		double [][] result = getResults(parameters);
		double error = 0;
		int errorCnt = 0;
		for (int i = 0; i < parameters.getIterationNum(); i++) {
			debugPrint("Iteration "+(i+1));
			error = 0;
			//1. optimize the DNN one by one
			if (true) {
				for (int j = 0; j < input.length; j++) {
					error = error + runEpoch(input[j], j, result[j], parameters);
				}
			}	
			//2. optimize the DNN over all
			errorCnt++;
			if (errorCnt % 1 == 0) {
				info("Fine tuning error ="+error);
			}			
			if (error < acc) {
				System.out.println("Training is stopped early.");
				break;				
			}
		}
 	}

}
