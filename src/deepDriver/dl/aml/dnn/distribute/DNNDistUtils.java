package deepDriver.dl.aml.dnn.distribute;

import deepDriver.dl.aml.ann.ArtifactNeuroNetwork;
import deepDriver.dl.aml.ann.ILayer;
import deepDriver.dl.aml.ann.INeuroUnit;
import deepDriver.dl.aml.ann.InputParameters;

public class DNNDistUtils {
	
	public static void copyWws(double [][] wWs, ArtifactNeuroNetwork ann, boolean copy2Ann) {
		int layerNum = 0;
		int neuNum = 0;
		
		ILayer layer = ann.getFirstLayer();
		layerNum = 0; 
		while (layer.getNextLayer() != null) {
			int preNum = layer.getNeuros().size();
			layer = layer.getNextLayer();
			
			int cln = layerNum ++;
			int wWsNum = 0;
			if (!copy2Ann) {
				wWs[cln] = new double[layer.getNeuros().size() * (preNum + 1)];//b is there.
			}
			
			for (int i = 0; i < layer.getNeuros().size(); i++) {
				INeuroUnit nu = layer.getNeuros().get(i);
				double [] thetas = nu.getThetas();
				for (int j = 0; j < thetas.length; j++) {
					if (copy2Ann) {
						thetas[j] = wWs[cln][wWsNum++];
					} else {
						wWs[cln][wWsNum++] = thetas[j];
					}					
				}
			}
		}
	}
	
	
	public static InputParameters [] splitTasks(double [][] results, int cnt, InputParameters parameters) {
//		 = rm.getClientsNum();
//		 = ann.getResults(parameters);

		InputParameters [] tasks = new InputParameters[cnt];
		int nilen = parameters.getInput().length/cnt;
		int rslen = results.length/cnt;
		
		for (int i = 0; i < tasks.length; i++) {
			tasks[i] = new InputParameters();
			tasks[i].setAlpha(parameters.getAlpha());
			tasks[i].setLamda(parameters.getLamda());
			tasks[i].setM(parameters.getM());
			if (parameters.getInput() == null) {//Input is null, so we can not caculate further more.
				continue;
			}
			double [][] dni = null;
			double [][] drs = null;
			
			if (i == tasks.length - 1) {
				dni = new double[parameters.getInput().length - i * nilen][];
				drs = new double[results.length - i * rslen][];
			} else {
				dni = new double[nilen][];
				drs = new double[rslen][];
			}			
			
			
			for (int j = 0; j < dni.length; j++) {
				double [] tmd = parameters.getInput()[i*nilen + j];
				dni[j] = tmd;
//				dni[j] = new double[tmd.length];
//				for (int k = 0; k < tmd.length; k++) {
//					dni[j][k] = tmd[k];
//				}
			}
			
			for (int j = 0; j < drs.length; j++) {
				double [] tmd = results[i*rslen + j];
				drs[j] = tmd;
//				drs[j] = new double[tmd.length];
//				for (int k = 0; k < tmd.length; k++) {
//					drs[j][k] = tmd[k];
//				}
			}
			
			tasks[i].setInput(dni);
			tasks[i].setResult2(drs);			
		}
		return tasks;
	}

}
