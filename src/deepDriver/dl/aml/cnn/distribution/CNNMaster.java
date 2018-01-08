package deepDriver.dl.aml.cnn.distribution;

import java.io.Serializable;

import deepDriver.dl.aml.cnn.CNNParaMerger;
import deepDriver.dl.aml.cnn.ConvolutionNeuroNetwork;
import deepDriver.dl.aml.cnn.IDataStream;
import deepDriver.dl.aml.common.distribution.CommonSlave;
import deepDriver.dl.aml.distribution.Error;
import deepDriver.dl.aml.distribution.ResourceMaster;

public class CNNMaster implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public boolean isSetup() {
		return ResourceMaster.getInstance().isSetup();
	}
	
	public void trainModel(IDataStream is, IDataStream tis, ConvolutionNeuroNetwork cnn) {
		ResourceMaster rm = ResourceMaster.getInstance();
		int cnt = rm.getClientsNum();
		IDataStream [] ids = null;
		/****/
		ids = is.splitStream(cnt); 
		int nilen = is.splitCnt(cnt);
		DataStreamDistUtil du = new DataStreamDistUtil();
		try {
			System.out.println("Distribute model name");
			rm.distributeCommand(CommonSlave.CMODEL_SLAVE+"="+CNNSlave.class.getName());
		} catch (Exception e1) {
			e1.printStackTrace();
		}		
		
		if (ids == null) {
			try {
				du.distributeDs(is, cnt);
			} catch (Exception e) { 
				e.printStackTrace();
			}
			nilen = (int) ((double)du.getCnt()/(double)cnt);
		}
		
		
		double [][] wWs = null;
		double err = 0;
		boolean firstDist = true;		
		double acc = cnn.getCfg().getAcc();
		int i = 0;
		int loop = 0;
		
		while (firstDist || err > acc) {
			err = 0;			
			for (int k = 0; k < nilen/CNNSlave.mb + 1; k++) {
				loop++;
				Object[] objs = null;
				try {
					if (firstDist) {
						objs = rm.run(ids, cnn);
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
				System.out.println("Complete "+ k * CNNSlave.mb+" samples, the error is "+err);
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
			System.out.println("Iteration "+(i++)+", error is " + err/(double)(nilen * cnt));
			cnnMerger.merge(cnn, wWs, true);
			try {
				cnn.saveCfg2File(cnn.getCfg().getName() + "-"+ loop, cnn.getCfg());
			} catch (Exception e) { 
				e.printStackTrace();
			} 
		}
		 
		
	}
	
	CNNParaMerger cnnMerger = new CNNParaMerger();	
	public void copy2Matrix(double [][] source, double [][] copy2, double len) {
		for (int i = 0; i < copy2.length; i++) {
			if (source[i] == null) {
				continue;
			}
			if (copy2[i] == null) {
				copy2[i] = new double[source[i].length];
			}			
			for (int j = 0; j < copy2[i].length; j++) {
				copy2[i][j] = copy2[i][j] + source[i][j]/len;
			}
		}
	}

}
