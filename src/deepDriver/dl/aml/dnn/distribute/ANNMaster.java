package deepDriver.dl.aml.dnn.distribute;

import java.io.Serializable;

import deepDriver.dl.aml.ann.ArtifactNeuroNetwork;
import deepDriver.dl.aml.ann.InputParameters;
import deepDriver.dl.aml.common.distribution.CommonSlave;
import deepDriver.dl.aml.distribution.Error;
import deepDriver.dl.aml.distribution.ResourceMaster;
import deepDriver.dl.aml.lstm.distribution.LSTMSlave;

public class ANNMaster implements Serializable {
	
	private static final long serialVersionUID = 1L;
	ArtifactNeuroNetwork ann;
//	ResourceMaster rm = ResourceMaster.getInstance();
	
	public boolean isSetup() {
		return ResourceMaster.getInstance().isSetup();
	}
		
	
	public void trainModel(ArtifactNeuroNetwork ann, InputParameters parameters, double acc) {
		ResourceMaster rm = ResourceMaster.getInstance();
		int cnt = rm.getClientsNum();
		InputParameters [] tasks = DNNDistUtils.splitTasks(ann.getResults(parameters), cnt, parameters);
		int nilen = parameters.getInput().length/cnt;
		
		double [][] wWs = null;
		double err = 0;
		boolean firstDist = true;
		
		try {
			System.out.println("Distribute model name");
			rm.distributeCommand(CommonSlave.CMODEL_SLAVE+"="+ANNSlave.class.getName());
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		
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
		
		DNNDistUtils.copyWws(wWs, ann, true);
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
		ANNMaster dm = new ANNMaster();
		double [][] b = new double[a.length][];
		dm.copy2Matrix(a, b, 2);
		dm.copy2Matrix(a1, b, 2);
		
	}
	
}
 