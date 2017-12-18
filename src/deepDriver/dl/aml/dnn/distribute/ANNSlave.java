package deepDriver.dl.aml.dnn.distribute;

import deepDriver.dl.aml.ann.ArtifactNeuroNetwork;
import deepDriver.dl.aml.ann.ILayer;
import deepDriver.dl.aml.ann.InputParameters;
import deepDriver.dl.aml.distribution.Error;
import deepDriver.dl.aml.distribution.Slave;
import deepDriver.dl.aml.dnn.DNN;

public class ANNSlave extends Slave {
	
	InputParameters parameters;
	ArtifactNeuroNetwork ann;
	
	double [][] input;
	@Override
	public void setTask(Object obj) throws Exception {
		System.out.println("batch size is: "+mb);
		if (obj instanceof String) {
			String ss = (String) obj;
			if (DNNMaster.Task_caculateHiddenInputs.equalsIgnoreCase(ss)) {
				if (dnn != null) {
					input = dnn.caculateHiddenInputs(parameters.getInput());
				}
			}			
		} else {
			parameters = (InputParameters) obj;
		}		
	}
	
//	double [][] wWs1;
//	double [][] wWs2;
	
	double [][] orig;
	double [][] curr;
	double [][] dwWs;

	int pos = 0;
	static int mb = 4096;
	Error errObj = new Error();
	double err = 0;
	@Override
	public void trainLocal() throws Exception {
//		parameters.setIterationNum(1);
//		parameters.setAlpha(1.0);
//		parameters.setM(0);
//		ann.trainModel(parameters);		
		double [][] input = parameters.getInput();
		double [][] result = ann.getResults(parameters);
		err = 0; 
		for (int i = 0; i < mb; i++) {	
			if (pos > input.length - 1) {
				pos = 0;
			}
			err = err + ann.runEpoch(input[pos], pos, result[pos], parameters);
			pos++;
		}
	}
	
	
	@Override
	public Error getError() {
		errObj.setErr(err);
		return errObj;
	}

	DNN dnn;
	@Override
	public void setSubject(Object obj) {		
		if (obj instanceof ArtifactNeuroNetwork) {
			ann = (ArtifactNeuroNetwork) obj;	
			if (obj instanceof DNN) {
				dnn = (DNN) obj;				
			}
		} else {
			orig = (double[][]) obj;
			DNNDistUtils.copyWws(orig, ann, true);
		}		
//		int layerNum = getLayerNum();		
//		orig = new double[layerNum][];		
//		copyWws(orig, ann); 
	}
	
	public int getLayerNum() {
		ILayer layer = ann.getFirstLayer();
		int layerNum = 0;
		while (layer.getNextLayer() != null) {
			layer = layer.getNextLayer();
			layerNum ++;
		}
		return layerNum;
	}

	@Override
	public Object getLocalSubject() {
		int layerNum = getLayerNum();		
		curr = new double[layerNum][];		
		DNNDistUtils.copyWws(curr, ann, false);
		return curr;
//		dwWs = new double[curr.length][];
//		for (int i = 0; i < curr.length; i++) {
//			dwWs[i] = new double[curr[i].length];
//			for (int j = 0; j < curr[i].length; j++) {
//				dwWs[i][j] = curr[i][j] - orig[i][j];
//			}
//		}
//		
//		return dwWs;
	}

}
