package deepDriver.dl.aml.lstm.attentionEnDecoder;

import java.io.File;
import java.util.List;


import deepDriver.dl.aml.distribution.Fs;
import deepDriver.dl.aml.lstm.IStream;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMDataSet;
import deepDriver.dl.aml.lstm.PosValue;
import deepDriver.dl.aml.lstm.beamSearch.BeamLayer;
import deepDriver.dl.aml.lstm.beamSearch.BeamNode;
import deepDriver.dl.aml.lstm.beamSearch.BeamSearch;

public class AttentionEnDecoderLSTM {
	AttentionEnDecoderBPTT encoderDecoderBPTT;
	IStream qis;
	IStream ais;
	
	LSTMConfigurator qcfg;
	LSTMConfigurator acfg;	
	
	AttentionCfg attCfg;
	
	public AttentionEnDecoderLSTM(LSTMConfigurator qcfg, LSTMConfigurator acfg) {
		super();
		this.qcfg = qcfg;
		this.acfg = acfg;			
	}

	public void trainModel(IStream qis, IStream ais, boolean skip) {
		this.qis = qis;
		this.ais = ais;
		encoderDecoderBPTT = new AttentionEnDecoderBPTT(qcfg, acfg);
		attCfg = new AttentionCfg(qcfg, acfg, encoderDecoderBPTT.attention);
		System.out.println("Begin to train the model.");
		while (true && !skip) {
			double err = runEpich();
			if (cnt > 0 && acfg.getAccuracy() > 0 && acfg.getAccuracy() > err) {
				break;
			}
		}
	}
	
	int cnt = 0;
	public double runEpich() {
		double error = 0;		
		qis.reset();
		ais.reset();
		boolean first = true;
		double lastAvg = 0;

		while (ais.hasNext()) {
			ais.next();
			Object pos = ais.getPos();
			qis.next(pos);
			cnt++;
			double m = acfg.getM();
			if (first) {
				System.out.println("m="+m+", l="+acfg.getLearningRate());
				first = false;
			}
			if (acfg.getLr() != null) {
				double l = acfg.getLr().adjustML(error, acfg.getLearningRate());
				if (l != acfg.getLearningRate()) {
					acfg.setLearningRate(l);
					acfg.setM(0);
					System.out.println("m="+0+", l="+l);
					qcfg.setLearningRate(l);
					qcfg.setM(0);
				}				
			}			
			error = error + encoderDecoderBPTT.runEpich(qis.getSampleTT(), 
					qis.getTarget(), ais.getSampleTT(), ais.getTarget());
			acfg.setM(m);
			qcfg.setM(m);
			double avgErr = error/(double)cnt;
			if (cnt % 200 == 0) {
				System.out.println(acfg.getName() + " avg Error is: "+ avgErr +" with "+cnt+", lastAvg: "+lastAvg);
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
//		save2File(middleName, qcfg);
//		save2File(middleName, acfg);
		save2File(middleName, attCfg, attCfg.getName());
		
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
	
	public void switchPv(PosValue s, PosValue t) {
		t.setPos(s.getPos());
		t.setValue(s.getValue());
	}
	
	public PosValue [] constructPosValueArr(double [] ts) {
		PosValue [] pvs = new PosValue[ts.length];
		for (int i = 0; i < pvs.length; i++) {
			pvs[i] = new PosValue();
			pvs[i].setPos(i);
			pvs[i].setValue(ts[i]);
		}
		PosValue tpv = new PosValue();
		for (int i = 0; i < pvs.length; i++) {
			for (int j = i + 1; j < pvs.length; j++) {
				if (pvs[i].getValue() < pvs[j].getValue()) {
					switchPv(pvs[i], tpv);
					switchPv(pvs[j], pvs[i]);
					switchPv(tpv, pvs[j]);
				}
			}			
		}
		return pvs;
	}
	
	
	public BeamSearch beamSearch(LSTMDataSet qds, int bmSize, LSTMDataSet ads, int plen, double[] eofChar) {
		encoderDecoderBPTT = new AttentionEnDecoderBPTT(qcfg, acfg);
		encoderDecoderBPTT.fTTEncoder(qds.getSamples()[0]);
		
		double[][] samples = ads.getSamples()[0];
		double[][] targets = null;
		BeamSearch bs = new BeamSearch();
		bs.setSize(bmSize); 
		
		targets = encoderDecoderBPTT.fTTDecoder(samples);
		PosValue[] pvs = constructPosValueArr(targets[targets.length - 1]);
		PosValue[] pvs1 = new PosValue[bs.getSize()];
		for (int i = 0; i < pvs1.length; i++) {
			pvs1[i] = pvs[i];
		}
		bs.createBn(pvs1, null);
		bs.sortAndPrunchByLastLayer();	
		plen = plen - 1;
		for (int i = 0; i < plen; i++) {
			BeamLayer bl = bs.getLayers().get(bs.getLayers().size() - 1);
			List<BeamNode> bns = bl.getBns();
			for (int j = 0; j < bns.size(); j++) {
				BeamNode parent = bns.get(j);
				double [][] v1 = bs.getBnById(j, bl);
				double [][] v = new double[v1.length + samples.length][1];
				int cnt = 0;
				for (int k = 0; k < samples.length; k++) {
					v[cnt++] = samples[k];
				}
				for (int k = 0; k < v1.length; k++) {
					v[cnt++] = v1[k];
				}
				targets = encoderDecoderBPTT.fTTDecoder(v);
				PosValue [] pvs2 = constructPosValueArr(targets[targets.length - 1]);
				PosValue [] pvs3 = new PosValue[1];
				for (int k = 0; k < pvs3.length; k++) {
					pvs3[k] = pvs2[k];
				}
				bs.createBn(pvs3, parent);
			}			
			bs.sortAndPrunchByLastLayer();			
		}

		return bs;
	}

}
