package deepDriver.dl.aml.lstm.distribution;


import deepDriver.dl.aml.distribution.AsycSlave;
import deepDriver.dl.aml.distribution.Error;
import deepDriver.dl.aml.lstm.LSTM;
import deepDriver.dl.aml.lstm.LSTMCfgCleaner;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMDeltaWwFromWwUpdater;
import deepDriver.dl.aml.lstm.LSTMWwUpdater;
import deepDriver.dl.aml.lstm.LSTMXwWUpdater;
import deepDriver.dl.aml.lstm.Seq2SeqLSTM;

public class Seq2SeqAsycSlaveV6 extends AsycSlave {
	Seq2SeqLSTM seq2SeqLSTM;
	SimpleTask task;
	Seq2SeqLSTMBoostrapper boot;
	public Seq2SeqAsycSlaveV6(Seq2SeqLSTMBoostrapper boot) {
		super();
		this.boot = boot;
	}
	int trainRound = 0;
	LSTMConfigurator cfgFromSrv;
//	LSTMWwUpdater deltaWwUdt = new LSTMWwUpdater(false, false);
	LSTMWwUpdater WwUpdater = new LSTMWwUpdater();
	LSTMWwUpdater deltaWwUpdater = new LSTMWwUpdater(false, false);
	boolean testQ = true;
//	private Seq2SeqLSTMConfigurator cfgFromSrv;
	
	LSTMCfgCleaner lSTMCfgCleaner = new LSTMCfgCleaner();
	
	Verifier verifier = new Verifier(false);
	public void setSubject(Object obj) {
		cfgFromSrv = (LSTMConfigurator) obj;
		testQ = Seq2SeqMasterV2.isQMode(cfgFromSrv);
		seq2SeqLSTM.getCfg().setTestQ(testQ);
		System.out.println("Srv Q mode="+ testQ+", " +
				"batchSize4DeltaWw="+
				cfgFromSrv.isBatchSize4DeltaWw() +
						", round"+cfgFromSrv.getLoop());
		
//		testWw(cfgFromSrv.getQlSTMConfigurator());
		
//		lSTMCfgCleaner.clean(seq2SeqLSTM.getCfg());
//		seq2SeqLSTM.setTrainCfg(cfgFromSrv, testQ);
//		lSTMCfgCleaner.gbClean();
//		testWw(cfgFromSrv.getQlSTMConfigurator());
		LSTMConfigurator cfg =  seq2SeqLSTM.getCfg().getQlSTMConfigurator();
		if (testQ) {			
		} else {
			cfg = seq2SeqLSTM.getCfg().getAlSTMConfigurator();
		}		
		copyWw(cfgFromSrv, cfg);
//		xwWUpdater.setResetDeltaWw(true);
//		xwWUpdater.updatewWs(cfgFromSrv, cfg);  
		out(cfg);
		verifier.verifyDistributeWws(cfg);
//		System.out.println("Already fresh Ww from server" );
	}
	
	LSTMXwWUpdater xwWUpdater = new LSTMXwWUpdater();
	
	private void out(LSTMConfigurator cfg) {
		System.out.println("M="+cfg.getM()+"," +
				"bs4deltaWw="+cfg.isBatchSize4DeltaWw()+"," +
				"l="+cfg.getLearningRate()+", " +
				"bs="+cfg.getMiniBatchSize()+","+
				"measureOnly="+cfg.isMeasureOnly());
	}
	
	public void copyWw(LSTMConfigurator fcfg, LSTMConfigurator t2cfg) {
		WwUpdater.updatewWs(fcfg
				, t2cfg);
		/*
		 * do we need the delta ww from server, 
		 * or we just need to keep the delta ww locally
		 * let us try with it yes
		 * The dynamic m should be initialize always. 
		 * ***/
		deltaWwUpdater.updatewWs(fcfg, t2cfg); 
		t2cfg.setM(fcfg.getM());
		t2cfg.setLearningRate(fcfg.getLearningRate());
		t2cfg.setBatchSize4DeltaWw(fcfg.isBatchSize4DeltaWw());
		t2cfg.setMeasureOnly(fcfg.isMeasureOnly());
	}
	
	LSTMWwUpdater wWchecker = new LSTMWwUpdater(true, true);
	public void testWw(LSTMConfigurator fcfg) {
		System.out.println("Prepare to check Wws");
		wWchecker.updatewWs(fcfg, seq2SeqLSTM.getCfg().getQlSTMConfigurator());
		System.out.println("Done to check Wws");
	}
	
	private int getloopCoverage(int taskSize, int mbsize) {
		int loop = taskSize/mbsize;
		if (loop * mbsize == taskSize) {
			return loop;
		}
		return loop+1;
	}
	
	
	protected int getLoopNum(int taskSize, int mbsize) {
		int loop = getloopCoverage(taskSize, mbsize);
		int [] loops = new int[] {
				getloopCoverage(taskSize - 1, mbsize),
				getloopCoverage(taskSize + 1, mbsize)
				};
		for (int i = 0; i < loops.length; i++) {
			if (loop < loops[i]) {
				loop = loops[i];
			}
		}
		return loop;
	}
	
	long l = 0;

	
	public void trainLocal() throws Exception {		
		System.out.println("Train locally with in Q mode ? "+seq2SeqLSTM.getCfg().isTestQ());
		LSTMConfigurator cfg = seq2SeqLSTM.getCfg().getQlSTMConfigurator();
		LSTM clstm = seq2SeqLSTM.getQlstm();
		if (!seq2SeqLSTM.getCfg().isTestQ()) {
			cfg = seq2SeqLSTM.getCfg().getAlSTMConfigurator();
			clstm = seq2SeqLSTM.getAlstm();
		}
		if (l <= 0) {
			l = System.currentTimeMillis();
		}
		trainRound ++;
		int taskSize = (task.getEnd() - task.getStart() + 1);	
		cfg.setMBSize(task.getMbatch());
		cfg.setInteractiveUpdate(true);
		
		seq2SeqLSTM.trainModel(false, boot.getQsi(), boot.getAsi(), boot.getNna());
		
		int loop = getLoopNum(taskSize, task.getMbatch());
		
		if (trainRound == loop) {
			System.out.println("It costs "+(System.currentTimeMillis() - l)+" to finish this round.");
//			cfg.setMBSize(taskSize - task.getMbatch() * (loop - 1));
			trainRound = 0;
			clstm.finish1Cycle();
			l = System.currentTimeMillis();
		} else {
			
		}
		System.out.println("Complete with batch size "+task.getMbatch()+", client loop "+trainRound+"/"+loop);
	}
	
	LSTMDeltaWwFromWwUpdater batchDeltaWwUpdaterV2 = new LSTMDeltaWwFromWwUpdater();
	
	@Override
	public Object getLocalSubject() {
//		System.out.println("Prepare the local cfg.." );
		LSTMConfigurator cfg =  seq2SeqLSTM.getCfg().getQlSTMConfigurator();
		if (seq2SeqLSTM.getCfg().isTestQ()) {
		} else {
			cfg = seq2SeqLSTM.getCfg().getAlSTMConfigurator();
		}	
		verifier.cfgClientWws(cfg);
		
//		copyWw(cfg, cfgFromSrv);
		batchDeltaWwUpdaterV2.mergeDeltawWs(new LSTMConfigurator[] {cfg}, cfgFromSrv); 
		
		System.out.println("Get the local cfg ready" );		
		return cfgFromSrv;
	}
	
	boolean bootstrapped = false;

	public void prepareData(boolean isServer) throws Exception {
		boot.prepareData(isServer);
	}
	
	public void setTask(Object obj)  throws Exception {
//		System.out.println("Receive task assigned from server");
		task = (SimpleTask) obj;
//		seq2SeqLSTM.getCfg().getQlSTMConfigurator().setMBSize(task.getMbatch());
//		seq2SeqLSTM.getCfg().getAlSTMConfigurator().setMBSize(task.getMbatch());
		System.out.println("The task is: "+task.getStart()+"/"+task.getEnd());
		if (bootstrapped) {
			return;
		}
		bootstrapped = true;
		boot.bootstrap(task, false);
		this.seq2SeqLSTM = boot.getSeq2SeqLSTM();
		System.out.println("Finished with LSTM bootstrap..");
	}

	public Error getError() {
//		System.out.println("prepare the Error");
		Error err = new Error();
		if (seq2SeqLSTM.getCfg().isTestQ()) {
			err.setReady(seq2SeqLSTM.getQlstm().isFinish1Cycle());
			err.setCnt(seq2SeqLSTM.getQlstm().getCnt());
			err.setErr(seq2SeqLSTM.getQlstm().getError());
		} else {
			err.setReady(seq2SeqLSTM.getAlstm().isFinish1Cycle());
			err.setCnt(seq2SeqLSTM.getAlstm().getCnt());
			err.setErr(seq2SeqLSTM.getAlstm().getError());
		}		
		System.out.println("Error is ready");
		return err;
	}
}
