package deepDriver.dl.aml.lstm.distribution;

import java.io.File;


import deepDriver.dl.aml.distribution.Error;
import deepDriver.dl.aml.distribution.Fs;
import deepDriver.dl.aml.lstm.GradientNormalizer;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMDeltaWwFromWwUpdater;
import deepDriver.dl.aml.lstm.Seq2SeqLSTMConfigurator;
import deepDriver.dl.aml.lstm.data.LSTMCfgData;

public class Seq2SeqMasterV5 extends Seq2SeqMaster {
	boolean measureOnly = false;
	public Seq2SeqMasterV5(int clientsNum, Seq2SeqLSTMBoostrapper boot,
			String sfile) throws Exception {
		super(clientsNum, boot);
		if (sfile != null) {
			System.out.println("Start with mfile->"+sfile);
			Seq2SeqLSTMConfigurator cfg = (Seq2SeqLSTMConfigurator) Fs.readObjFromFile(sfile);
			wWUpdater.updatewWs(cfg.getQlSTMConfigurator(), srvQ2QLSTM.getCfg().getQlSTMConfigurator());
			wWUpdater.updatewWs(cfg.getAlSTMConfigurator(), 
					srvQ2QLSTM.getCfg().getAlSTMConfigurator());

			this.srvQ2QLSTM.setTrainCfg(srvQ2QLSTM.getCfg());
		}
		srvQ2QLSTM.getCfg().getQlSTMConfigurator().setMeasureOnly(measureOnly);
		srvQ2QLSTM.getCfg().getAlSTMConfigurator().setMeasureOnly(measureOnly);
	}
	
	public Seq2SeqMasterV5(int clientsNum, Seq2SeqLSTMBoostrapper boot) throws Exception {
		super(clientsNum, boot);
	}
	
	public static int QType = 1;
	public static int AType = 2;	
	LSTMConfigurator cfgFromClient;
		
	public static boolean isQMode(LSTMCfgData lstmCfgFromClient) {
		if (lstmCfgFromClient.getType() == QType) {
			return true;
		}
		return false;
	}
	
	public static boolean isQMode(LSTMConfigurator lstmCfgFromClient) {
		if (lstmCfgFromClient.getType() == QType) {
			return true;
		}
		return false;
	}
	
	Verifier verifier = new Verifier(false);
	
	public Object getDistributeSubject() {
//		System.out.println("Prepare srv cfg to clients"+srvQ2QLSTM.getCfg().getLoop());
		LSTMConfigurator cfg = srvQ2QLSTM.getCfg().getQlSTMConfigurator();
		int type = QType;
		if (srvQ2QLSTM.getCfg().isTestQ()) {
		} else {
			cfg = srvQ2QLSTM.getCfg().getAlSTMConfigurator();
			type = AType;
		}
		
		verifier.cfgDistributeWws(cfg);
		
		if (cfgFromClient == null) {
			cfgFromClient = cfg;			
		} else {
			copyWws(cfg, cfgFromClient);
		}/*****/
		cfgFromClient.setType(type);
		cfgFromClient.setLoop(srvQ2QLSTM.getCfg().getLoop());
		int loop = tq/batchSize;		
		System.out.println("Ready for srv cfg with "+cfgFromClient.getLoop()+" rounds");
		return cfgFromClient;
	}
	
	public void copyWws(LSTMConfigurator fcfg, LSTMConfigurator t2cfg) {
		wWUpdater.updatewWs(fcfg, t2cfg);	
		deltaWwUpdater.setResetDeltaWw(true);
		deltaWwUpdater.updatewWs(fcfg, t2cfg);
		t2cfg.setM(fcfg.getM());
		t2cfg.setLearningRate(fcfg.getLearningRate());
//		t2cfg.setM(0);
//		t2cfg.setLearningRate(1);
		t2cfg.setBatchSize4DeltaWw(fcfg.isBatchSize4DeltaWw());
		t2cfg.setMeasureOnly(fcfg.isMeasureOnly());
	}
	
	GradientNormalizer gn = new GradientNormalizer();
	
	double threshold = 50;
	
	double dl = 1;
	int lCycle = 3;
	protected double getL(LSTMConfigurator cfg) {
		double nl = cfg.getLearningRate();
		int a = epichNum / lCycle;
		nl = dl;
		if (a > 0) {
			double d = Math.pow(2, a);
//			nl = dl/(5.0 * (double)(a));
			nl = dl/d;
		}		
		System.out.println("" +"Learn with M"+cfg.getM()+", L"+nl);
		return nl;
	}
	
	LSTMDeltaWwFromWwUpdater batchDeltaWwUpdaterV2 = new LSTMDeltaWwFromWwUpdater();

	public void mergeSubject(Object[] objs) {
//		System.out.println("Prepare to merge Ww from clients");
		LSTMConfigurator[] cfgs = new LSTMConfigurator[objs.length];
		for (int i = 0; i < cfgs.length; i++) {
			cfgs[i] = (LSTMConfigurator) objs[i]; 
		}
		srvQ2QLSTM.getCfg().setLoop(srvQ2QLSTM.getCfg().getLoop() + 1); 
		LSTMConfigurator cfg = srvQ2QLSTM.getCfg().getQlSTMConfigurator();

		if (srvQ2QLSTM.getCfg().isTestQ()) {			
		} else {
			cfg = srvQ2QLSTM.getCfg().getAlSTMConfigurator();
		}	
		
//		double l = getL(cfg);
		
//		batchDeltaWwUpdaterV2.mergeDeltawWs(cfgs, cfg, l, cfg.getM());
		batchDeltaWwUpdaterV2.mergeDeltawWs(cfgs, cfg);
		gn.normGradient(cfg, threshold);
		lSTMWwFresher.freshwWs(cfg);
				
		if (cfgFromClient != srvQ2QLSTM.getCfg().getQlSTMConfigurator()
				&& cfgFromClient != srvQ2QLSTM.getCfg().getAlSTMConfigurator()) {
			lSTMCfgCleaner.clean(cfgFromClient);
		}		
		cfgFromClient = cfgs[0];
		cfgs[0] = null; 
		for (int i = 1; i < cfgs.length; i++) {
			lSTMCfgCleaner.clean(cfgs[i]);
			cfgs[i] = null;
		}
		lSTMCfgCleaner.gbClean();
		
		verifier.verifyMergeWws(cfg);
	}
	

	double lastError = 0;
	boolean isM = false; 
	
	boolean neverAdjust = false;
	double m = 0;
	
	private void adjustML(double err, LSTMConfigurator cfg) {
		if (neverAdjust) {
			return ;
		}
//		if (m == 0) {
//			m = cfg.getM();
//		}
//		cfg.setM(m);
		if (lastError > 0 && err > lastError) {
//			cfg.setM(0);			
			if (cfg.getM() > 0 && isM) {
//				cfg.m = cfg.m / 3.0 * 2.0;				
				cfg.setM(cfg.getM() / 3.0 * 2.0);
				System.out.println("Adjust M");
				System.out.println("Distribute M"+cfg.getM()+", L"+cfg.getLearningRate());

			} else {
//				cfg.learningRate = cfg.learningRate / 3.0 * 2.0;
				if (cfg.getLearningRate() > 0.0001) {}
				cfg.setLearningRate(cfg.getLearningRate()/ 3.0 * 2.0);
								
				System.out.println("Adjust L");
				System.out.println("Distribute M"+cfg.getM()+", L"+cfg.getLearningRate());

			}
			//do not adjust M
//			isM = !isM;		
		}
		lastError = err;
	}
	
	int epichNum = 0;
	
	public double caculateErrorLastTime(Object[] objs) {
		double err = 0;
		double cnt = 0;		
		Error ce1 = (Error)objs[0];
		for (int i = 0; i < objs.length; i++) {
//			System.out.println("Prepare to merge client "+index
//				+", run "+error.getCnt()+" samples, with avg error "+avgErr);
		}
		
		if (!ce1.isReady()) {
			return 0;
		}
		epichNum ++;
		for (int i = 0; i < objs.length; i++) {
			Error ce = (Error)objs[i];
//			if (!ce.isReady()) {
//				return 0;
//			}
			err = err + ce.getErr();
			cnt = cnt + 1;//+ ce.getCnt();
		}
		err = err/cnt;
		
		if (srvQ2QLSTM.getCfg().isTestQ()) {
			if (err <= srvQ2QLSTM.getCfg().getQlSTMConfigurator().getAccuracy()) {
				srvQ2QLSTM.getCfg().setTestQ(false);
				System.out.println("Switch to A mode, and reset the cfg");
				cfgFromClient = null;
				lastError = 0;
				isM = false;
				System.out.println("Complete Q lstm training.");
				save2File();
				epichNum = 1; 
			} else {
				adjustML(err, srvQ2QLSTM.getCfg().getQlSTMConfigurator());
			}
			System.out.println(cnt+" is tested Q with error "+ err);
		} else {
			if (err <= srvQ2QLSTM.getCfg().getAlSTMConfigurator().getAccuracy()) {
				this.done = true;
				save2File();
				System.out.println("Complete A lstm training");
			} else {
				adjustML(err, srvQ2QLSTM.getCfg().getAlSTMConfigurator());
			}
			System.out.println(cnt+" is tested A with error "+ err);
		}				
		return err;
	}
	
	String cfgFileName = "seq2seqCfg";
	private void save2File() {
		String sf = System.getProperty("user.dir");
		long a = System.currentTimeMillis();
		File dir = new File(sf, "data");
		dir.mkdirs();
		File f = new File(dir, cfgFileName+"_"+a+".m");
		try {
			Fs.writeObj2FileWithTs(f.getAbsolutePath(), srvQ2QLSTM.getCfg());
			System.out.println("Save "+cfgFileName+" into "+f.getAbsolutePath());
		} catch (Exception e) {
			e.printStackTrace();
		}		
	}
}
