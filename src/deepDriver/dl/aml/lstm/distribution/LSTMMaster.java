package deepDriver.dl.aml.lstm.distribution;

import java.io.Serializable;

import deepDriver.dl.aml.distribution.Error;
import deepDriver.dl.aml.distribution.ResourceMaster;
import deepDriver.dl.aml.lstm.IStream;
import deepDriver.dl.aml.lstm.LSTM;
import deepDriver.dl.aml.lstm.LSTMWwArrayTranslator;

public class LSTMMaster implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public boolean isSetup() {
		return ResourceMaster.getInstance().isSetup();
	}
	
	public void trainModel(IStream is, IStream tis, LSTM lstm) {
		ResourceMaster rm = ResourceMaster.getInstance();
		int cnt = rm.getClientsNum();
		IStream [] ids = is.splitStream(cnt); 
		int nilen = is.splitCnt(cnt);
		double [][] wWs = null;
		double err = 0;
		boolean firstDist = true;		
		double acc = lstm.getCfg().getAccuracy();
		int i = 0;
		int loop = 0;
		while (firstDist || err > acc) {
			err = 0;			
			for (int k = 0; k < nilen/LSTMSlave.mb + 1; k++) {
				loop++;
				Object[] objs = null;
				try {
					if (firstDist) {
						objs = rm.run(ids, lstm);
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
				System.out.println("Complete "+ k * LSTMSlave.mb+" samples, the error is "+err);
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
//			cnnMerger.merge(lstm, wWs, true);
			translator.update(lstm.getCfg(), wWs, true);
			try {
				lstm.save2File(lstm.getCfg().getName() + "-"+ loop);
			} catch (Exception e) { 
				e.printStackTrace();
			} 
		}
		 
		
	}
	
	LSTMWwArrayTranslator translator = new LSTMWwArrayTranslator();
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
