package deepDriver.dl.aml.lstm;

import java.io.Serializable;

import deepDriver.dl.aml.lrate.StepReductionLR;


public class LSTMConfigurator implements Serializable {
/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	int type;
	int loop;
	
	//	RNNNeuroVo [][] rnnVos;
	IRNNLayer [] layers;
//	//Time--layer---nodePosition
//	double [][][] aATT;
//	double [][][] zZTT;
//	//layer--nodePosition---weight
//	double [][][] wWTL;	
//	IStream is;
	transient IPreCxtProvider preCxtProvider;
	transient NeuroNetworkArchitecture nna;
	int maxTimePeriod = 40;
	
	transient ICxtConsumer cxtConsumer;	
	
	String name;	
	
	boolean bp4FirstLayer = false;
	
	boolean measureOnly = false;
	
	boolean batchSize4DeltaWw = true;	
	
	boolean useThinData = false;	
	
	boolean useBias = false;	
	
	boolean debug = false;	
	
	int threadsNum = 1;	
	
	boolean biDirection = false;
			
	public boolean isBiDirection() {
		return biDirection;
	}

	public void setBiDirection(boolean biDirection) {
		this.biDirection = biDirection;
	}

	public boolean isBp4FirstLayer() {
		return bp4FirstLayer;
	}

	public void setBp4FirstLayer(boolean bp4FirstLayer) {
		this.bp4FirstLayer = bp4FirstLayer;
	}

	public int getThreadsNum() {
		return threadsNum;
	}

	public void setThreadsNum(int threadsNum) {
		this.threadsNum = threadsNum;
	}

	public boolean isDebug() {
		return debug;
	}

	public void setDebug(boolean debug) {
		this.debug = debug;
	}

	public boolean isUseBias() {
		return useBias;
	}

	public void setUseBias(boolean useBias) {
		this.useBias = useBias;
	}

	public boolean isUseThinData() {
		return useThinData;
	}
	
	public void setUseThinData(boolean useThinData) {
		this.useThinData = useThinData;
	}

	public boolean isMeasureOnly() {
		return measureOnly;
	}

	public void setMeasureOnly(boolean measureOnly) {
		this.measureOnly = measureOnly;
	}

	public boolean isBatchSize4DeltaWw() {
		return batchSize4DeltaWw;
	}

	public void setBatchSize4DeltaWw(boolean batchSize4DeltaWw) {
		this.batchSize4DeltaWw = batchSize4DeltaWw;
	}

	public IRNNLayer[] getLayers() {
		return layers;
	}

	public void setLayers(IRNNLayer[] layers) {
		this.layers = layers;
	}

	public int getType() {
		return type;
	}

	public void setType(int type) {
		this.type = type;
	}

	public int getLoop() {
		return loop;
	}

	public void setLoop(int loop) {
		this.loop = loop;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public IPreCxtProvider getPreCxtProvider() {
		return preCxtProvider;
	}

	public void setPreCxtProvider(IPreCxtProvider preCxtProvider) {
		this.preCxtProvider = preCxtProvider;
	}

	public int getMaxTimePeriod() {
		return maxTimePeriod;
	}

	public void setMaxTimePeriod(int maxTimePeriod) {
		this.maxTimePeriod = maxTimePeriod;
	}
	
	boolean requireLastRNNLayer = true;

	public boolean isRequireLastRNNLayer() {
		return requireLastRNNLayer;
	}

	public void setRequireLastRNNLayer(boolean requireLastRNNLayer) {
		this.requireLastRNNLayer = requireLastRNNLayer;
	}
	
	public void buildArchitecture(IStream is, NeuroNetworkArchitecture nna) {
		buildArchitecture(is.getTargetFeatureNum(), is.getSampleFeatureNum(), nna);
	}

	public void buildArchitecture(int targetFeatureNum, int sampleFeatureNum, NeuroNetworkArchitecture nna) {
//		double [][][] samples = ds.getSamples();
//		double [][] sample = samples[0];
//		timePeriod = sample.length;
//		timePeriod = ds.getSampleTTLength();
//		double [] sampleFeature = sample[0];
//		double [] sampleTarget = ds.getTargets()[0][0];
		this.costFunction = nna.getCostFunction();
		this.nna = nna;
//		cfg.buildArchitecture(is, nna);
		int layerNum = nna.getNnArch().length + 2;
		if (!requireLastRNNLayer) {
			layerNum = nna.getNnArch().length + 1;
		}
//		if (nna.isUseProjectionLayer()) {
//			layerNum = layerNum + 1;
//		}
		layers = new IRNNLayer[layerNum];
//		rnnVos = new RNNNeuroVo[layerNum][];
		for (int i = 0; i < layers.length; i++) {
			int nextNN = 0;
			if (i < layers.length - 2) {
				nextNN = nna.getNnArch()[i];
			} else {
				nextNN = targetFeatureNum;
			}
			if (i == 0) {
//				rnnVos[i] = new RNNNeuroVo[sampleFeature.length];
				layers[i] = createRNNLayer(sampleFeatureNum, maxTimePeriod, false, 0, nextNN);
			} else if (i == ProjectionLayerID && nna.isUseProjectionLayer()) {
				layers[i] = createProjectionLayer(nna.getNnArch()[i - 1], maxTimePeriod, nextNN);
			} else if (i == layers.length - 1 && requireLastRNNLayer) {
//				rnnVos[i] = new RNNNeuroVo[sampleTarget.length];
//				constructRNNNeuroVos(rnnVos[i], timePeriod, true, rnnVos[i - 1].length, 
//						rnnVos[i].length);
				layers[i] = createRNNLayer(targetFeatureNum, maxTimePeriod, false, layers[i-1].getRNNNeuroVos().length, 0);
			} else {
//				rnnVos[i] = new RNNNeuroVo[nna.getNnArch()[i - 1]];
//				constructRNNNeuroVos(rnnVos[i], timePeriod, false, rnnVos[i - 1].length, 
//						rnnVos[i].length);
				if (nna.getHiddenType() == NeuroNetworkArchitecture.HiddenRNN) {
					layers[i] = createRNNLayer(nna.getNnArch()[i - 1], maxTimePeriod, true, layers[i-1].getRNNNeuroVos().length, nextNN);
				} else {
					layers[i] = createLSTMLayer(nna.getNnArch()[i - 1], maxTimePeriod, true, layers[i-1].getRNNNeuroVos().length, nextNN);
				}
			}	
		}
	}
	
	static int ProjectionLayerID = 1;
	
	public IRNNLayer createProjectionLayer(int projectionLength, int maxT, int nextLayerNN) {
		return new ProjectionLayer(projectionLength, maxT, nextLayerNN);
	}
	
	public IRNNLayer createLSTMLayer(int nodeNN, 
			int t, boolean inHidenLayer, int previousNNN, int nextLayerNN) {
		if (biDirection) {
			return new BiLstmLayer(nodeNN, 
					t, inHidenLayer, previousNNN, nextLayerNN);
		}
		return new LSTMLayerV2(nodeNN, 
				t, inHidenLayer, previousNNN, nextLayerNN);
	}
	
	public IRNNLayer createRNNLayer(int nodeNN, 
			int t, boolean inHidenLayer, int previousNNN, int nextLayerNN) {
		if (biDirection && previousNNN == 0) {
			return new BiRNNLayer(nodeNN, 
					t, inHidenLayer, previousNNN, nextLayerNN);
		}
		return new RNNLayer(nodeNN, 
				t, inHidenLayer, previousNNN, nextLayerNN);
	}
//	public void buildArchitecture(LSTMDataSet ds, NeuroNetworkArchitecture nna) {
//		this.ds = ds;
//		this.nna = nna;
//		double [][][] samples = ds.getSamples();
//		double [][] sample = samples[0];
//		double [] sampleFeature = sample[0];
//		double [] sampleTarget = ds.getTargets()[0][0];
//		aATT = new double[sample.length][][];
//		int layerNum = nna.getNnArch().length + 2;
//		for (int i = 0; i < aATT.length; i++) {
//			aATT[i] = new double[layerNum][];
//			for (int j = 0; j < aATT[i].length; j++) {
//				if (j == 0) {
//					aATT[i][j] = new double[sampleFeature.length];
//				} else if (j == aATT[i].length - 1) {
//					aATT[i][j] = new double[sampleTarget.length];
//				} else {
//					aATT[i][j] = new double[nna.getNnArch()[j - 1]];
//				}				
//			}
//		}
//		
//	}
	
//	public void constructZz() {
//		zZTT = new double[aATT.length][][];
//		for (int i = 0; i < zZTT.length; i++) {
//			zZTT[i] = new double[aATT[i].length][];
//			for (int j = 0; j < zZTT[i].length; j++) {
//				zZTT[i][j] = aATT[i][j];
//			}
//		}
//	}
	public static int LEAST_SQUARE = 1;
	public static int SOFT_MAX = 2;
	int costFunction = LEAST_SQUARE;
	int loopNum = 5;
	
	public int getLoopNum() {
		return loopNum;
	}

	public void setLoopNum(int loopNum) {
		this.loopNum = loopNum;
	}
	
	double accuracy = -1.0; 
	
	public double getAccuracy() {
		return accuracy;
	}

	public void setAccuracy(double accuracy) {
		this.accuracy = accuracy;
	}
	
	int miniBatchSize = 1;	
	
	boolean interactiveUpdate = false;	
	
	boolean useRmsProp = false;	
	
	public boolean isUseRmsProp() {
		return useRmsProp;
	}

	public void setUseRmsProp(boolean useRmsProp) {
		this.useRmsProp = useRmsProp;
	}

	public boolean isInteractiveUpdate() {
		return interactiveUpdate;
	}

	public void setInteractiveUpdate(boolean interactiveUpdate) {
		this.interactiveUpdate = interactiveUpdate;
	}

	boolean enableUseCellAa = true;	
	
	
	public boolean isEnableUseCellAa() {
		return enableUseCellAa;
	}

	public void setEnableUseCellAa(boolean enableUseCellAa) {
		this.enableUseCellAa = enableUseCellAa;
	}

	boolean binaryLearning = false;
	
	boolean useRandomResult = false;
	

	public boolean isUseRandomResult() {
		return useRandomResult;
	}

	public void setUseRandomResult(boolean useRandomResult) {
		this.useRandomResult = useRandomResult;
	}

	public boolean isBinaryLearning() {
		return binaryLearning;
	}

	public void setBinaryLearning(boolean binaryLearning) {
		this.binaryLearning = binaryLearning;
	}

	public int getMiniBatchSize() {
		return miniBatchSize;
	}

	public void setMBSize(int miniBatchSize) {
		this.miniBatchSize = miniBatchSize;
	}
	
	public ICxtConsumer getCxtConsumer() {
		return cxtConsumer;
	}

	public void setCxtConsumer(ICxtConsumer cxtConsumer) {
		this.cxtConsumer = cxtConsumer;
	}
	
	double learningRate = 0.01;
	double m = 0.8;
	double dropOut = 0;	
	boolean forceComplete = false;	
	
	public boolean isForceComplete() {
		return forceComplete;
	}

	public void setForceComplete(boolean forceComplete) {
		this.forceComplete = forceComplete;
	}

	public double getDropOut() {
		return dropOut;
	}

	public void setDropOut(double dropOut) {
		this.dropOut = dropOut;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public double getM() {
		return m;
	}

	public void setM(double m) {
		this.m = m;
	}

	StepReductionLR lr;

	public StepReductionLR getLr() {
		return lr;
	}

	public void setLr(StepReductionLR lr) {
		this.lr = lr;
	}
	
	StepReductionLR srm;

	public StepReductionLR getSrm() {
		return srm;
	}

	public void setSrm(StepReductionLR srm) {
		this.srm = srm;
	}	

}
