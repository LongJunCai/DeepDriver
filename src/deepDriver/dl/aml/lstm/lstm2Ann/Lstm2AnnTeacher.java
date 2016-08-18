package deepDriver.dl.aml.lstm.lstm2Ann;

import java.io.File;

import deepDriver.dl.aml.ann.ANN;
import deepDriver.dl.aml.distribution.Fs;
import deepDriver.dl.aml.lstm.BPTT;
import deepDriver.dl.aml.lstm.IStream;
import deepDriver.dl.aml.lstm.LSTMConfigurator;

public class Lstm2AnnTeacher {
	LSTMConfigurator qcfg;
	ANN ann;
	
	BPTT qBptt;
	Lstm2AnnBPTT lstm2AnnBPTT;
	IStream qis;
	
	public Lstm2AnnTeacher(LSTMConfigurator qcfg, ANN ann) {
		super();
		this.qcfg = qcfg;
		this.ann = ann;
		lstm2AnnBPTT = new Lstm2AnnBPTT(qcfg, ann);
	}
	
	public void trainModel(IStream qis, boolean skip) {
		this.qis = qis;
		System.out.println("Begin to train the model.");
		while (true && !skip) {
			double err = runEpich();
			save2File(rcnt++ % 10 +"");
			if (cnt > 0 && qcfg.getAccuracy() > 0 && qcfg.getAccuracy() > err) {
				break;
			}
		}
	}
	
	int cnt = 0;
	public double runEpich() {
		double error = 0;		
		qis.reset(); 
		boolean first = true;
		double lastAvg = 0;

		while (qis.hasNext()) {
			qis.next();
			Object pos = qis.getPos();
			qis.next(pos);
			cnt++;
			double m = qcfg.getM();
			if (qcfg.getLr() != null) {
				double l = qcfg.getLr().adjustML(error, qcfg.getLearningRate());
				if (l != qcfg.getLearningRate()) {
					qcfg.setLearningRate(l);
					qcfg.setM(0);
					System.out.println("m="+0+", l="+l);
					qcfg.setLearningRate(l);
					qcfg.setM(0);
				}				
			}			
			error = error + lstm2AnnBPTT.runEpich(qis.getSampleTT(), 
					qis.getTarget()); 
			qcfg.setM(m);
			double avgErr = error/(double)cnt;
			if (cnt % 200 == 0) {
				System.out.println(qcfg.getName() + " avg Error is: "+ avgErr +" with "+cnt);
				if (avgErr < 50.0) {
					if (lastAvg == 0 || lastAvg -  avgErr> 1.0) {
						save2File(rcnt++ % 10 +"");
						lastAvg = avgErr;
					}					
					first = false;
				}
			}
			
		}
		return error / (double) cnt;
	}
	int rcnt = 0;
	
	long currentTimestamp = System.currentTimeMillis();
	private void save2File(String middleName) {
		save2File(middleName, qcfg, qcfg.getName());
		save2File(middleName, ann, ann.getName());
	}
	private void save2File(String middleName, Object obj, String name) {
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
			Fs.writeObj2FileWithTs(f.getAbsolutePath(), obj);
			System.out.println("Save "+name+" into "+f.getAbsolutePath());
		} catch (Exception e) {
			e.printStackTrace();
		}		
	}
	

}
