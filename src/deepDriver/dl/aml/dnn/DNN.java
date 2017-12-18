package deepDriver.dl.aml.dnn;


import deepDriver.dl.aml.ann.ArtifactNeuroNetwork;
import deepDriver.dl.aml.ann.ArtifactNeuroNetworkV2;
import deepDriver.dl.aml.ann.ILayer;
import deepDriver.dl.aml.ann.InputParameters;
import deepDriver.dl.aml.ann.SparseAutoEncoder;
import deepDriver.dl.aml.ann.imp.LayerImpV2;
import deepDriver.dl.aml.dnn.distribute.ANNMaster;
import deepDriver.dl.aml.dnn.distribute.DNNMaster;

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
	
	DNNMaster dnnMaster = new DNNMaster();	
	public void distributeTask(InputParameters orignalPramameters) {
		if (dnnMaster.isSetup()) {
			try {
				dnnMaster.distributeTasks(this, orignalPramameters);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}		
	}
	
	public void distExpendLayer() {
		if (dnnMaster.isSetup()) {
			try {
				dnnMaster.expendLayer();
				} catch (Exception e) {
				e.printStackTrace();
			}
		}		
	}
	
	public void distPreTraining() {
		if (dnnMaster.isSetup()) {
			try {
				dnnMaster.preTraining();
				} catch (Exception e) {
				e.printStackTrace();
			}
		}		
	}
	
	public void distFineTuning() {
		if (dnnMaster.isSetup()) {
			try {
				dnnMaster.distFineTuning();
				} catch (Exception e) {
				e.printStackTrace();
			}
		}		
	}
		
	public void learnSelf(InputParameters orignalPramameters) {
		double [][] input = orignalPramameters.getInput();
		if (normalize) {
			input = normalizer.transformParameters(orignalPramameters.getInput());
		}		
		int ln = orignalPramameters.getLayerNum();
		firstLayer = createLayerOnly();
		slLamda = orignalPramameters.getLamda();
		this.setFirstLayer(firstLayer);
		debugPrint("Begin to build up the DNN, start to pre-training first");
		
		double [][] tmpInp = new double[][]{input[0]};
		firstLayer.buildup(null, tmpInp, createAcf(), false, input[0].length);
//		ILayer currentLayer = firstLayer;
		boolean sl4LastLayer = true;
		if (sl4LastLayer) {
		} else {
			ln = ln - 1;
		}		
		/**add Distribution supporting.***/
		distributeTask(orignalPramameters);
		/*****/
		/**add Distribution supporting.***/
		distPreTraining();
		/*****/
		for (int i = 0; i < ln; i++) {
			int hiddenNc = input[0].length;
			if (orignalPramameters.getNeuros() != null) {
				hiddenNc = orignalPramameters.getNeuros()[i];
			}
			System.out.println("Pre-training for layer "+(i + 1));
			double [][] newInput = caculateHiddenInputs(input);
			ILayer last= getLastLayer();
			InputParameters newPramaters = new InputParameters();
			
			newPramaters.setAlpha(orignalPramameters.getAlpha());
			newPramaters.setLamda(slLamda);
			
			newPramaters.setIterationNum(slLoopNum);
//			int nc = newInput[0].length;
			int nc = last.getNeuros().size();
			if (newInput == null) {
				newInput = new double[][]{new double[nc]};
			}
			newPramaters.setInput(newInput);
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
//			ILayer newHiddenLayer = sae.getFirstLayer().getNextLayer();		
//			newHiddenLayer.setPos(i+1);
//			last.setNextLayer(newHiddenLayer);
//			newHiddenLayer.setNextLayer(null);
//			newHiddenLayer.setPreviousLayer(last);			
//			LayerImpV2 l2 = (LayerImpV2) newHiddenLayer;
//			l2.setaNNCfg(getaNNCfg());
			expendLayerFromSAE(sae, i, last);
			/**add Distribution supporting.***/
			distExpendLayer();
			/*****/
		}
		if (!sl4LastLayer) {
			ILayer last= getLastLayer();
			ILayer newLayer = createLayer();
			newLayer.setPos(orignalPramameters.getLayerNum());
			newLayer.buildup(last, input, createAcf(), 
				true, orignalPramameters.getNeuros()[orignalPramameters.getNeuros().length - 1]);
		}
		/**add Distribution supporting.***/
		distFineTuning();
		/*****/
	}
	
	public void expendLayerFromSAE(ArtifactNeuroNetwork ann, int i, ILayer last) {
		ILayer newHiddenLayer = ann.getFirstLayer().getNextLayer();		
		newHiddenLayer.setPos(i+1);
		last.setNextLayer(newHiddenLayer);
		newHiddenLayer.setNextLayer(null);
		newHiddenLayer.setPreviousLayer(last);			
		LayerImpV2 l2 = (LayerImpV2) newHiddenLayer;
		l2.setaNNCfg(getaNNCfg());
	}
	
	
	public double [][] caculateHiddenInputs(double [][] orginalInput) {	
		/**add Distribution supporting.***/
		if (dnnMaster.isSetup()) {
			try {
				dnnMaster.caculateHiddenInputs();
			} catch (Exception e) {
				e.printStackTrace();
			}
			return null;
		}
		/*****/
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
	

	public double getAcc() {
		return acc;
	}

	public void setAcc(double acc) {
		this.acc = acc;
	}
	double acc = 0.1;
	
	transient ANNMaster dm = new ANNMaster();
	public void tuneFine(InputParameters parameters) {
		
		if (dm != null && dm.isSetup()) {
			System.out.println("DNN Runing in the distribution env.");
			dm.trainModel(this, parameters, acc);
			return;
		}
		System.out.println("DNN Runing in the standalone env.");
				
		System.out.println("Prepare for fine tuning.");
		double [][] input = parameters.getInput();
		if (normalize) {
			input = normalizer.transformParameters(parameters.getInput());
		}		
				
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
