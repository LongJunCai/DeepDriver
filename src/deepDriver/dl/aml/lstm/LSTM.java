package deepDriver.dl.aml.lstm;

import java.io.File;
import java.util.List;
import java.util.Random;

import deepDriver.dl.aml.distribution.Fs;
import deepDriver.dl.aml.lstm.beamSearch.BeamLayer;
import deepDriver.dl.aml.lstm.beamSearch.BeamNode;
import deepDriver.dl.aml.lstm.beamSearch.BeamSearch;

public class LSTM {
	
	LSTMConfigurator cfg;
	public void setCfg(LSTMConfigurator cfg) {
		this.cfg = cfg;
		if (this.bPTT != null) {
			this.bPTT.cfg = cfg;
		}		
	}
	
	public void rebuild(LSTMConfigurator cfg) {
		this.cfg = cfg; 
		this.bPTT = createBPTT();
	}

	public LSTM(LSTMConfigurator cfg) {
		super();
		this.cfg = cfg;
	}
	
	IStream is;
	
	ITest phaseTest;	

	public ITest getPhaseTest() {
		return phaseTest;
	}

	public void setPhaseTest(ITest phaseTest) {
		this.phaseTest = phaseTest;
	}
	
	GradientNormalizer gNormalizer = new GradientNormalizer();
	
	public void trainModel(IStream is) {
		trainModel(is, false);
	}

	public void trainModel(IStream is, boolean skip) {
		double lastError = 0;		
		this.is = is;		
		bPTT = createBPTT();
		boolean isM = false;
		int cnt = 0;
		long st = System.currentTimeMillis();
		double threshold = 2;
		while (true && !skip) {
			gNormalizer.normGradient(cfg, threshold);
			if (cfg.forceComplete) {
				System.out.println("Force to complete.");
				break;
			}
			if (cfg.accuracy < 0 && cnt > cfg.loopNum) {
				break;
			}
			if (cnt > 0 && cfg.accuracy > 0 && cfg.accuracy > lastError) {
				break;
			}
			System.out.println(cfg.name + " the " + (cnt + 1) + " round starts:");
			double error = runEpich();
			if (cnt > 0 && error > lastError) {
				if (isM) {
					cfg.m = cfg.m / 3.0 * 2.0;
				} else {
					cfg.learningRate = cfg.learningRate / 3.0 * 2.0;
				}
				isM = !isM;
			}
			double changRate = (lastError - error) / lastError;
			lastError = error;
			System.out.println(cfg.name + " the error is: " + error
					+ ", and change rate is: " + changRate);
			
			save2File(cnt % 10 +"");
			
			long t1 = System.currentTimeMillis();
			System.out.println(cfg.name + " costed: " + (t1 - st));
			st = t1;
			cnt++;
			if (phaseTest != null) {
				try {
					phaseTest.test();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
		for (int i = 0; i < cfg.loopNum; i++) {
		}
	}
	
	long currentTimestamp = System.currentTimeMillis();
	
	private void save2File(String middleName) {
		String sf = System.getProperty("user.dir");		
		File dir = new File(sf, "data");
		dir.mkdirs();		
		File f = null;
		if (middleName == null) {
			f = new File(dir, cfg.getName()+"_"+currentTimestamp+".m");
		} else {
			f = new File(dir, cfg.getName()+"_"+currentTimestamp+
					"_"+middleName+".m");
		}		
		try {
			Fs.writeObj2FileWithTs(f.getAbsolutePath(), cfg);
			System.out.println("Save "+cfg.getName()+" into "+f.getAbsolutePath());
		} catch (Exception e) {
			e.printStackTrace();
		}		
	}
	
	public boolean isFinish1Cycle() {
		return finish1Cycle;
	}
	
	public void finish1Cycle() {
		finish1Cycle = true;
		error = error/cnt;
		System.out.println("Finish 1 cycle, reset the error and cnt");
	}

	public void setFinish1Cycle(boolean finish1Cycle) {
		this.finish1Cycle = finish1Cycle;		
	}
	
	public BPTT createBPTT() {
		if (cfg.isBiDirection()) {
			return new BiBPTT(cfg);
		} else {
			return new BPTT4MultThreads(cfg);
		}
	}
	
	public double trainModelWithBatchSize(IStream is) {
		if (this.is != is) {				
			this.is = is;		
			bPTT = createBPTT();					
			is.reset();
			if (cfg.preCxtProvider != null) {
				cfg.preCxtProvider.reset();
			}
		}  		
		if (finish1Cycle) {
			finish1Cycle = false;
			cnt = 0;
			error = 0;
		}
		int loop = cfg.getMiniBatchSize();
		long st = System.currentTimeMillis();
		for (int i = 0; i < loop; i++) {			
			if (!is.hasNext()) {
				is.reset();
				if (cfg.preCxtProvider != null) {
					cfg.preCxtProvider.reset();
				}
				if (cfg.cxtConsumer != null) {
					cfg.cxtConsumer.complete();
				}
			}
			is.next();
			cnt++;
			error = error + runEpich(is.getSampleTT(), is.getTarget());
		}	
		System.out.println("Complete cnt="+cnt+", bsize="+cfg.getMiniBatchSize()
				+", avg error="+ (error/(double)cnt)+", time="+(System.currentTimeMillis() - st));		
		return error;
	}

	boolean finish1Cycle = false;
	double error = 0;
	int cnt = 0;
	double lastError = 0;
	boolean isM = false; 
	public double trainModelWithBatchSize2(IStream is) {
		if (this.is != is) {				
			this.is = is;		
			bPTT = createBPTT();			
			long st = System.currentTimeMillis();
			if (cfg.preCxtProvider != null) {
				cfg.preCxtProvider.reset();
			}
		} 		
		if (finish1Cycle) {
			finish1Cycle = false;
			cnt = 0;
			error = 0;
			if (cfg.preCxtProvider != null) {
				cfg.preCxtProvider.reset();
			}
		}
		
		int loop = cfg.getMiniBatchSize();
//		cfg.setMBSize(1);
		for (int i = 0; i < loop; i++) {			
//			if (cfg.forceComplete) {
//				break;
//			}
			if (!is.hasNext()) {
				cfg.setMBSize(loop);
				finish1Cycle = true;
				is.reset();
				evaluatePhasely();	
				break;
			}
			is.next();
			cnt++;
			error = error + runEpich(is.getSampleTT(), is.getTarget());
		}
		if (!is.hasNext()) {
			cfg.setMBSize(loop);
			finish1Cycle = true;
			is.reset();
			evaluatePhasely();	
		} 			

		System.out.println("Complete batch size "+cfg.getMiniBatchSize()
				+" with error "+ error);		
		return error;
	}
		
	public void evaluatePhasely() {
		error = error / (double) cnt;
		lastError = error;		
//		if (lastError > 0 && error > lastError) {
//			if (isM) {
//				cfg.m = cfg.m / 3.0 * 2.0;
//			} else {
//				cfg.learningRate = cfg.learningRate / 3.0 * 2.0;
//			}
//			isM = !isM;
//		}
		System.out.println("Complete the whole round, with "+cnt
				+" samples, with error "+ error);
		if (cfg.cxtConsumer != null) {
			cfg.cxtConsumer.complete();
		}
	}	

	public int getCnt() {
		return cnt;
	}

	public void setCnt(int cnt) {
		this.cnt = cnt;
	}

	public double getError() {
		return error;
	}

	public void setError(double error) {
		this.error = error;
	}
	
	public double runEpich() {
		double error = 0;
		int cnt = 0;
		is.reset();
		if (cfg.preCxtProvider != null) {
			cfg.preCxtProvider.reset();
		}

		while (is.hasNext()) {
			if (cfg.forceComplete) {
				break;
			}
			is.next();
			cnt++;
			double m = cfg.getM();
			if (cfg.lr != null) {
				double l = cfg.lr.adjustML(error, cfg.getLearningRate());
				if (l != cfg.getLearningRate()) {
					cfg.setLearningRate(l);
					cfg.setM(0);
					System.out.println("m="+0+", l="+l);
				}				
			}			
			error = error + runEpich(is.getSampleTT(), is.getTarget());
			cfg.setM(m);
//			if (cfg.isDebug() && error < cfg.getAccuracy()) {
//				break;
//			}
			
			if (cnt % 200 == 0) {
				System.out.println(cfg.getName() + " avg Error is: "+ error/(double)cnt+" with "+cnt);
			}
		}
		if (cfg.cxtConsumer != null) {
			cfg.cxtConsumer.complete();
		}
		return error / (double) cnt;
		// double [][][] samples = ds.getSamples();
		// double [][][] targets = ds.getTargets();
		// for (int i = 0; i < samples.length; i++) {
		// runEpich(samples[i], targets[i]);
		// }
	}

	BPTT bPTT = null;
	Object lockObj = new Object();

	public double runEpich(double[][] sample, double[][] targets) {
		if (cfg.preCxtProvider != null) {
			if (cfg.preCxtProvider.hasNext()) {
				bPTT.setPreCxts(cfg.preCxtProvider.next());
			} else {
				if (cfg.preCxtProvider.isCompleted()) {
					return 0;
				} else {
					cfg.preCxtProvider.require(lockObj);
					if (cfg.preCxtProvider.hasNext()) {
						bPTT.setPreCxts(cfg.preCxtProvider.next());
					} else {
						return 0;
					}
				}
			}
		}
		double threshold = 5;
//		gNormalizer.normGradient(cfg, threshold);
		double err = bPTT.runEpich(sample, targets);
		destroyData(sample);
		destroyData(targets);
		if (cfg.cxtConsumer != null) {
			cfg.cxtConsumer.addContext(bPTT.getHLContext());
		}
		return err;
	}
	
	public double [][] test(double[][] sample, double[][] targets) {
//		if (cfg.preCxtProvider != null) {
//			if (cfg.preCxtProvider.hasNext()) {
//				bPTT.setPreCxts(cfg.preCxtProvider.next());
//			} else {
//				if (cfg.preCxtProvider.isCompleted()) {
//					return 0;
//				} else {
//					cfg.preCxtProvider.require(lockObj);
//					if (cfg.preCxtProvider.hasNext()) {
//						bPTT.setPreCxts(cfg.preCxtProvider.next());
//					} else {
//						return 0;
//					}
//				}
//			}
//		}
//		gNormalizer.normGradient(cfg, threshold);
		double [][] ts = bPTT.fTT(sample, true);
		destroyData(sample);
		destroyData(targets);
		if (cfg.cxtConsumer != null) {
			cfg.cxtConsumer.addContext(bPTT.getHLContext());
		}
		return ts;
	}
	
	boolean cleanData = true;	
	
	public boolean isCleanData() {
		return cleanData;
	}

	public void setCleanData(boolean cleanData) {
		this.cleanData = cleanData;
	}

	public void destroyData(double [][] arr) {
		if (!cleanData) {
			return;
		}
		for (int i = 0; i < arr.length; i++) {
			arr[i] = null;
		}
	}

	public double[][][] testModel(LSTMDataSet ds) {
		double[][][] samples = ds.getSamples();
		double[][][] targets = new double[samples.length][][];
		if (cfg.preCxtProvider != null) {
			if (cfg.preCxtProvider.hasNext()) {
				bPTT.setPreCxts(cfg.preCxtProvider.next());
			}
		}
		for (int i = 0; i < samples.length; i++) {
			targets[i] = bPTT.fTT(samples[i], true);
			targets[i] = convertTT(targets[i]);
		}
		if (cfg.cxtConsumer != null) {
			cfg.cxtConsumer.addContext(bPTT.getHLContext());
		}
		return targets;
	}

	public boolean compareChar(double target[], double s[]) {
		for (int i = 0; i < s.length; i++) {
			if (target[i] != s[i]) {
				return false;
			}
		}
		return true;
	}
	
	Random rd = new Random(System.currentTimeMillis());
//	class PosValue {
//		int pos;
//		double value;
//	}
	public PosValue [] constructPosValueArr(double [] ts) {
		PosValue [] pvs = new PosValue[ts.length];
		for (int i = 0; i < pvs.length; i++) {
			pvs[i] = new PosValue();
			pvs[i].pos = i;
			pvs[i].value = ts[i];
		}
		PosValue tpv = new PosValue();
		for (int i = 0; i < pvs.length; i++) {
			for (int j = i + 1; j < pvs.length; j++) {
				if (pvs[i].value < pvs[j].value) {
					switchPv(pvs[i], tpv);
					switchPv(pvs[j], pvs[i]);
					switchPv(tpv, pvs[j]);
				}
			}			
		}
		return pvs;
	}
	
	public void switchPv(PosValue s, PosValue t) {
		t.pos = s.pos;
		t.value = s.value;
	}
	
	public int getRandom(double [] ts) {
		double r = rd.nextDouble();
		double l = 0;
		int pos = 0;
		PosValue [] pvs = constructPosValueArr(ts);
		for (int i = 0; i < pvs.length; i++) {
			double nl = l + pvs[i].value;
			if (r >= l && r < nl) {
				pos = pvs[i].pos;
				break;
			}
			l = nl;
		}
		return pos;
	}

	public double[][] convertTT(double target[][]) {
		if (cfg.binaryLearning) {
			double[][] nt = new double[target.length][];
			for (int i = 0; i < nt.length; i++) {
				if (cfg.isUseThinData()) {
					nt[i] = new double[1];
				} else {
					nt[i] = new double[target[i].length];
				}				
				int pos = 0;
				double max = 0;
				for (int j = 0; j < target[i].length; j++) {
					if (max < target[i][j]) {
						max = target[i][j];
						pos = j;
					}
				}
				if (cfg.isUseRandomResult()) {
					pos = getRandom(target[i]);
				}		
				if (cfg.isUseThinData()) {
					nt[i][0] = pos;
				} else {
					nt[i][pos] = 1;
				}
				
			}
			return nt;
		} else {
			return target;
		}
	}

	public double[][][] testModel(LSTMDataSet ds, int plen, double[] eofChar) {
		return testModel(ds, plen, plen - 1, eofChar);
	}

	public double[][][] testModel(LSTMDataSet ds, int plen) {
		return testModel(ds, plen, plen - 1, null);
	}

	public double[][][] testModel(LSTMDataSet ds, int plen, int pdlen,
			double[] eofChar) {
		double[][][] samples = ds.getSamples();
		double[][][] targets = new double[samples.length][][];
		double[][][] nSamples = null;
		if (cfg.preCxtProvider != null) {
			if (cfg.preCxtProvider.hasNext()) {
				bPTT.setPreCxts(cfg.preCxtProvider.next());
			}
		}
		if (samples[samples.length - 1].length < cfg.maxTimePeriod) {
			nSamples = new double[samples.length][][];
		} else {
			nSamples = new double[samples.length + 1][][];
		}
		for (int i = 0; i < samples.length; i++) {
			nSamples[i] = samples[i];
			targets[i] = bPTT.fTT(samples[i], true);
			/****
			 * **/
			targets[i] = convertTT(targets[i]);
			int ll = targets[i].length;
			if (eofChar != null && compareChar(targets[i][ll - 1], eofChar)) {
				double[][][] nts = new double[i + 1][][];
				for (int j = 0; j < nts.length; j++) {
					nts[j] = targets[j];
				}
				return nts;
			}
			/****
			 * **/
		}
		if (cfg.cxtConsumer != null) {
			cfg.cxtConsumer.addContext(bPTT.getHLContext());
		}
		int j = targets.length - 1;
		if (nSamples.length > samples.length) {
			nSamples[nSamples.length - 1] = new double[1][samples[0][0].length];
			nSamples[nSamples.length - 1][0] = targets[j][targets[j].length - 1];
		} else {
			int lst = nSamples.length - 1;
			double[][] aa = nSamples[lst];
			nSamples[lst] = new double[aa.length + 1][];
			for (int i = 0; i < aa.length; i++) {
				nSamples[lst][i] = aa[i];
			}
			nSamples[lst][nSamples[lst].length - 1] = targets[j][targets[j].length - 1];
		}
		if (pdlen > 0) {
			ds.setSamples(nSamples);
			return testModel(ds, plen, pdlen - 1, eofChar);
		}
		return targets;
	}
	
	public BeamSearch beamSearch(int bmSize, LSTMDataSet ds, int plen, double[] eofChar) {
		double[][] samples = ds.getSamples()[0];
		double[][] targets = null;
		BeamSearch bs = new BeamSearch();
		bs.setSize(bmSize);
		
		if (cfg.preCxtProvider != null) {
			if (cfg.preCxtProvider.hasNext()) {
				bPTT.setPreCxts(cfg.preCxtProvider.next());
			}
		}
		
		targets = bPTT.fTT(samples, true);
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
				targets = bPTT.fTT(v, true);
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
	
	public boolean check(double [] ta, double [] tb) {
		if (getMaxPos(ta) == getMaxPos(tb)) {
			return true;
		}
		return false;
	}
	
	public int getMaxPos(double [] ta) {
		int pos = 0;
		for (int i = 0; i < ta.length; i++) {
			if (ta[i] > ta[pos]) {
				pos = i;
			}
		}
		return pos;
	}

	public void testModel(IStream qsi) {
		bPTT = createBPTT();	
		qsi.reset();
		int verifyNum = 0;
		int correctNum = 0;
		
		int scNum = 0;
		int correctScNum = 0;
		int exceedingNum = 0;
		while (qsi.hasNext()) {
			qsi.next();
			boolean scCorrect = true;
			scNum ++;			
			double [][] sample = qsi.getSampleTT();
			if (sample.length == 0) {
				continue;
			}
			double [][] ts = qsi.getTarget();
			double [][] rs = bPTT.fTT(sample, true);
//			System.out.println(ts.length +","+ rs.length);
			for (int i = 0; i < rs.length; i++) {
				verifyNum ++;
				if (i > ts.length - 1) {
					System.out.println(""+sample.length +", "+ts.length +", "+rs.length);
					exceedingNum ++;
				}
				if (i > ts.length - 1 || check(ts[i], rs[i])) {
					correctNum ++;
				} else {
					scCorrect = false;
				}
			}
			if (scCorrect) {
				correctScNum ++;
			}
		}
		System.out.println(verifyNum+" words in all, and "+correctNum
				+" words type are correct, the avg correction is: "+(double)correctNum/(double)verifyNum);
		System.out.println(scNum+" words in all, and "+correctScNum
				+" words type are correct, the avg correction is: "
				+(double)correctScNum/(double)scNum);
		System.out.println("Exceeding num is: "+ exceedingNum);
		
	}

}
