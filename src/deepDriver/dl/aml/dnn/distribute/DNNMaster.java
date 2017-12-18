package deepDriver.dl.aml.dnn.distribute;

import java.io.Serializable;

import deepDriver.dl.aml.ann.ArtifactNeuroNetwork;
import deepDriver.dl.aml.ann.InputParameters;
import deepDriver.dl.aml.distribution.Error;
import deepDriver.dl.aml.distribution.ResourceMaster;

public class DNNMaster implements Serializable {
	
	private static final long serialVersionUID = 1L;
	ArtifactNeuroNetwork ann;
	ResourceMaster rm = ResourceMaster.getInstance();
	
	public boolean isSetup() {
		return rm.isSetup();
	}
	
	public void trainModel(ArtifactNeuroNetwork ann, InputParameters parameters, double acc) {  
		int cnt = rm.getClientsNum();
		double [][] results = ann.getResults(parameters);
//		try {
//			Fs.writeObj2FileWithTs("D:\\6.workspace\\ANN\\parameters.cfg", parameters);
//		} catch (Exception e1) {
//			e1.printStackTrace();
//		}
		InputParameters [] tasks = new InputParameters[cnt];
		int nilen = parameters.getInput().length/cnt;
		int rslen = results.length/cnt;
		
		for (int i = 0; i < tasks.length; i++) {
			tasks[i] = new InputParameters();
			tasks[i].setAlpha(parameters.getAlpha());
			tasks[i].setLamda(parameters.getLamda());
			tasks[i].setM(parameters.getM());
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
				dni[j] = new double[tmd.length];
				for (int k = 0; k < tmd.length; k++) {
					dni[j][k] = tmd[k];
				}
			}
			
			for (int j = 0; j < drs.length; j++) {
				double [] tmd = results[i*rslen + j];
				drs[j] = new double[tmd.length];
				for (int k = 0; k < tmd.length; k++) {
					drs[j][k] = tmd[k];
				}
			}
			
			tasks[i].setInput(dni);
			tasks[i].setResult2(drs);			
		}
		
		double [][] wWs = null;
		double err = 0;
		boolean firstDist = true;
		for (int i = 0; i < parameters.getIterationNum(); i++) {
			err = 0;			
			for (int k = 0; k < nilen/DNNSlave.mb + 1; k++) {
				Object[] objs = null;
				try {
					if (firstDist) {
						objs = rm.run(tasks, ann);
						firstDist = false;
					} else {
						objs = rm.run(null, wWs);
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
				
				Object[] errs = rm.getErrs();
				
				for (int j = 0; j < errs.length; j++) {
					Error e = (Error) errs[j];
					err = err + e.getErr();
				}				
				double[][] dd = (double[][]) objs[0];
				wWs = new double[dd.length][];
				for (int j = 0; j < objs.length; j++) {
					double[][] dd1 = (double[][]) objs[j];
					copy2Matrix(dd1, wWs, objs.length);
				}
			}
			if (err < acc) {
				System.out.println("Training is stopped early.");
				break;				
			}
			System.out.println("Iteration "+i+", error is " + err/(double)(parameters.getInput().length));
		}
		
	}
	
	
	public void copy2Matrix(double [][] source, double [][] copy2, double len) {
		for (int i = 0; i < copy2.length; i++) {
			if (copy2[i] == null) {
				copy2[i] = new double[source[i].length];
			}			
			for (int j = 0; j < copy2[i].length; j++) {
				copy2[i][j] = copy2[i][j] + source[i][j]/len;
			}
		}
	}
	
	public static void main(String[] args) {
		double [][] a = new double[][]{{1,1},{2,2}};
		double [][] a1 = new double[][]{{3,3},{4,4}};
		DNNMaster dm = new DNNMaster();
		double [][] b = new double[a.length][];
		dm.copy2Matrix(a, b, 2);
		dm.copy2Matrix(a1, b, 2);
		
	}
	
}
 