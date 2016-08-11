package deepDriver.dl.aml.lstm.distribution;

import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMWwUpdater;

public class Seq2SeqSlaveV2 extends Seq2SeqSlave {

	public Seq2SeqSlaveV2(Seq2SeqLSTMBoostrapper boot) {
		super(boot);
	}
	
	LSTMConfigurator cfgFromSrv;
//	LSTMWwUpdater deltaWwUdt = new LSTMWwUpdater(false, false);
	
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
		out(cfg);
		verifier.verifyDistributeWws(cfg);
//		System.out.println("Already fresh Ww from server" );
	}
	
	private void out(LSTMConfigurator cfg) {
		System.out.println("M="+cfg.getM()+"," +
				"bs4deltaWw="+cfg.isBatchSize4DeltaWw()+"," +
				"l="+cfg.getLearningRate()+", " +
				"bs="+cfg.getMiniBatchSize());
	}
	
	public void copyWw(LSTMConfigurator fcfg, LSTMConfigurator t2cfg) {
		WwUpdater.updatewWs(fcfg
				, t2cfg);
		deltaWwUpdater.updatewWs(fcfg
					, t2cfg);
		t2cfg.setM(fcfg.getM());
		t2cfg.setLearningRate(fcfg.getLearningRate());
		t2cfg.setBatchSize4DeltaWw(fcfg.isBatchSize4DeltaWw());
	}
	
	LSTMWwUpdater wWchecker = new LSTMWwUpdater(true, true);
	public void testWw(LSTMConfigurator fcfg) {
		System.out.println("Prepare to check Wws");
		wWchecker.updatewWs(fcfg, seq2SeqLSTM.getCfg().getQlSTMConfigurator());
		System.out.println("Done to check Wws");
	}

	@Override
	public Object getLocalSubject() {
//		System.out.println("Prepare the local cfg.." );
		LSTMConfigurator cfg =  seq2SeqLSTM.getCfg().getQlSTMConfigurator();
		if (seq2SeqLSTM.getCfg().isTestQ()) {
		} else {
			cfg = seq2SeqLSTM.getCfg().getAlSTMConfigurator();
		}	
		verifier.cfgClientWws(cfg);
		
		copyWw(cfg, cfgFromSrv);
		
		System.out.println("Get the local cfg ready" );		
		return cfgFromSrv;
	}

}
