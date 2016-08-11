package deepDriver.dl.aml.cnn2lstm;

import java.io.File;


import deepDriver.dl.aml.cnn.CNNConfigurator;
import deepDriver.dl.aml.cnn.IDataMatrix;
import deepDriver.dl.aml.cnn.IDataStream;
import deepDriver.dl.aml.distribution.Fs;
import deepDriver.dl.aml.lstm.IStream;
import deepDriver.dl.aml.lstm.LSTMConfigurator;

public class CNN2LSTMTeacher {
	CNNConfigurator cnnCfg;
	LSTMConfigurator lstmCfg;
	
	CNN2LSTMBPTT cNN2LSTMBPTT;
	IDataStream cnnIs;
	IStream ais;

	public CNN2LSTMTeacher(CNNConfigurator cnnCfg, LSTMConfigurator lstmCfg) {
		super();
		this.cnnCfg = cnnCfg;
		this.lstmCfg = lstmCfg;		
	}
	
	public void trainModel(IDataStream cnnIs, IStream ais, boolean skip) {
		this.cnnIs = cnnIs;
		this.ais = ais;
		cNN2LSTMBPTT = new CNN2LSTMBPTT(cnnCfg, lstmCfg);
		System.out.println("Begin to train the CNN2LSTM model.");
		while (true && !skip) {
			double err = runEpich();
			if (cnt > 0 && lstmCfg.getAccuracy() > 0 && lstmCfg.getAccuracy() > err) {
				break;
			}
		}
	}
	
	int cnt = 0;
	public double runEpich() {
		double error = 0;		
		cnnIs.reset();
		ais.reset();
		boolean first = true;

		while (ais.hasNext()) {
			ais.next();
			Object pos = ais.getPos();
			IDataMatrix dm = cnnIs.next(pos);
			cnt++;
			double m = lstmCfg.getM();
			if (lstmCfg.getLr() != null) {
				double l = lstmCfg.getLr().adjustML(error, lstmCfg.getLearningRate());
				if (l != lstmCfg.getLearningRate()) {
					lstmCfg.setLearningRate(l);
					lstmCfg.setM(0);
					System.out.println("m="+0+", l="+l);
				}				
			}			
			error = error + cNN2LSTMBPTT.runEpich(ais.getSampleTT(), 
					ais.getTarget(), new IDataMatrix [] {dm}, dm.getTarget());
			lstmCfg.setM(m);
			double avgErr = error/(double)cnt;
			if (cnt % 200 == 0) {
				System.out.println(lstmCfg.getName() + " avg Error is: "+ avgErr +" with "+cnt);
			}
			if (avgErr < 30.0 && first) {
				save2File(rcnt++ % 10 +"");
				first = false;
			}
		}
		return error / (double) cnt;
	}
	int rcnt = 0;
	
	long currentTimestamp = System.currentTimeMillis();
	private void save2File(String middleName) {
		save2File(middleName, cnnCfg, cnnCfg.getName());
		save2File(middleName, lstmCfg, lstmCfg.getName());
	}
	private void save2File(String middleName, Object cfg, String name) {
		String sf = System.getProperty("user.dir");		
		File dir = new File(sf, "data");
		dir.mkdirs();		
		File f = null;
		if (middleName == null) {
			f = new File(dir, name+"_"+currentTimestamp+".m");
		} else {
			f = new File(dir, name+"_"+currentTimestamp+
					"_"+middleName+".m");
		}		
		try {
			Fs.writeObj2FileWithTs(f.getAbsolutePath(), cfg);
			System.out.println("Save "+name+" into "+f.getAbsolutePath());
		} catch (Exception e) {
			e.printStackTrace();
		}		
	}
	
	
	

}
