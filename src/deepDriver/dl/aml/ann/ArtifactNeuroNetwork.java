package deepDriver.dl.aml.ann;

import java.io.Serializable;

import deepDriver.dl.aml.ann.imp.LayerImp;
import deepDriver.dl.aml.ann.imp.LogicsticsActivationFunction;

public class ArtifactNeuroNetwork implements Serializable {
	
	private static final long serialVersionUID = 602126004810899386L;
	
	ILayer firstLayer;
	
	public ILayer createLayer() {
		return new LayerImp();
	}
	
	public IActivationFunction createAcf() {
		return new LogicsticsActivationFunction();
	}
	
	public ILayer getFirstLayer() {
		return firstLayer;
	}

	public void setFirstLayer(ILayer firstLayer) {
		this.firstLayer = firstLayer;
	}
	
	protected Normalizer normalizer = new Normalizer();
	boolean isIncrementalMode = true;//incremental or batch
	
	public double [][] getResults(InputParameters parameters) {
		double [][] result2 = parameters.getResult2();
		if (result2 != null) {
			return result2;
		}
		double [] result = parameters.getResult();
		double [][] newResult = new double[result.length][1];
		for (int i = 0; i < newResult.length; i++) {
			newResult[i][0] = result[i];
		}
		return newResult;
	}
	
	boolean useNormalizer = true;

	public boolean isUseNormalizer() {
		return useNormalizer;
	}

	public void setUseNormalizer(boolean useNormalizer) {
		this.useNormalizer = useNormalizer;
	}

	public void trainModel(InputParameters parameters) {
		//1.set value into first layer			
		double [][] input = parameters.getInput();
		if (useNormalizer) {
			input = normalizer.transformParameters(parameters.getInput());
		}
		double [][] result = getResults(parameters);
		IActivationFunction acf = createAcf();
		firstLayer = createLayer();
		debugPrint("Begin to build up the ann:");
		firstLayer.buildup(null, input, acf, false, input[0].length);
		ILayer tlayer = firstLayer;
		for (int i = 0; i < parameters.getLayerNum(); i++) {
			ILayer newLayer = createLayer();
			int neuroCnt = input[0].length;
			if (parameters.getNeuros() != null) {
				neuroCnt = parameters.getNeuros()[i];
			}
			newLayer.setPos(i+1);
			newLayer.buildup(tlayer, input, acf, 
					i == parameters.getLayerNum() - 1, neuroCnt);
			tlayer = newLayer;
		}
		debugPrint("Complete to build ann");
		//2.set value into first layer
		debugPrint("Begin training.");
		
		double error = 0;
		int errorCnt = 0;
		for (int i = 0; i < parameters.getIterationNum(); i++) {
			debugPrint("Iteration "+(i+1));
			error = 0;
			//1. optimize the ann one by one
			if (isIncrementalMode) {
				for (int j = 0; j < input.length; j++) {
					error = error + runEpoch(input[j], j, result[j], parameters);
				}
			} else {
				error = runEpoch(input, i, result, parameters);
			}			
			//2. optimize the ann over all
			errorCnt++;
			if (errorCnt % 1 == 0) {
				info("Error ="+error);
			}			
		}
		
	}
	
	public double runEpoch(double [][] input, int i, double [][] result, InputParameters parameters) {
		double old = 0;
		double newValue = -old;
//		double residual = old;
//		double precision = 0.00000000000001;
		ILayer layer = firstLayer;
		ILayer lastLayer = firstLayer;
		while (layer != null) {
			debugPrint("ForwardPropagation "+(i+1)+" on layer "+layer);
			layer.forwardPropagation(input);
			lastLayer = layer;
			layer = layer.getNextLayer();
		}		
		layer = lastLayer;
		newValue = lastLayer.getStdError(result);		
//		double tmpResidual = newValue - old;	
//		residual = tmpResidual;			
		old = newValue;
//		if(Double.isInfinite(residual) || residual < precision) {	
//			break;
//		}
		while (layer != null && firstLayer != layer) {
			debugPrint("BackPropagation "+(i+1)+" on layer "+layer);
			layer.backPropagation(result, parameters);
			lastLayer = layer;
			layer = layer.getPreviousLayer();
		}
		//update network
		layer = firstLayer;
		while (layer != null) {
			debugPrint("update layer "+(i+1)+" on layer "+layer);
			layer.updateNeuros();
			lastLayer = layer;
			layer = layer.getNextLayer();
		}
		return newValue;
	}
	
	public double runEpoch(double [] x, int i, double []y, InputParameters parameters) {
		double [][] result = new double[][]{y};
		double [][] input = new double[][]{x};
		return runEpoch(input, i, result, parameters);
	}
	
	public void debugPrint(String msg) {
		println(msg, DEBUG);
	}
	
	public void info(String msg) {
		println(msg, INFO);
	}
	
	static final int INFO = 2;
	static final int DEBUG = 1;
	static final int NOTHING = 100;
	int currentLevel = INFO;
	
	public void println(String msg, int level) {
		if (level >= currentLevel) {
			System.out.println(msg);
		}
	}
	
	public double [] testModel(InputParameters parameters) {
		ILayer lastLayer = firstLayer;
		ILayer layer = firstLayer;
		debugPrint("Begin to build up the ann for test:");
		double [][] input = normalizer.retransformParameters(parameters.getInput());
		firstLayer.buildup(null, input, null, false, input[0].length);
		while (layer != null) {
			debugPrint("ForwardPropagation on layer "+layer);
			layer.forwardPropagation(input);
			lastLayer = layer;
			layer = layer.getNextLayer();
		}
		double arr [] = new double[input.length];
		INeuroUnit neuro = lastLayer.getNeuros().get(0);
		for (int i = 0; i < arr.length; i++) {
			arr[i] = neuro.getAaz(i);
		}
		return arr;
	}
	
	public static void main(String[] args) {
		double [] a = {1,2};
		double [][] b = new double[][]{a};
		System.out.println(b.length +","+b[0].length);
	}
	
}
