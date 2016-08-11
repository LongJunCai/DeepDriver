package deepDriver.dl.aml.lstm.distribution;


import deepDriver.dl.aml.distribution.Error;
import deepDriver.dl.aml.distribution.Slave;
import deepDriver.dl.aml.lstm.LSTM;
import deepDriver.dl.aml.lstm.LSTMCfgCleaner;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMWwUpdater;
import deepDriver.dl.aml.lstm.Seq2SeqLSTM;
import deepDriver.dl.aml.lstm.Seq2SeqLSTMConfigurator;

public class Seq2SeqSlave extends Slave {
	Seq2SeqLSTM seq2SeqLSTM;
	SimpleTask task;
	Seq2SeqLSTMBoostrapper boot;
	
	public Seq2SeqSlave(Seq2SeqLSTMBoostrapper boot) {
		super();
		this.boot = boot;
	}

	boolean bootstrapped = false;
	@Override
	public void setTask(Object obj)  throws Exception {
//		System.out.println("Receive task assigned from server");
		task = (SimpleTask) obj;
//		seq2SeqLSTM.getCfg().getQlSTMConfigurator().setMBSize(task.getMbatch());
//		seq2SeqLSTM.getCfg().getAlSTMConfigurator().setMBSize(task.getMbatch());
		if (bootstrapped) {
			return;
		}
		bootstrapped = true;
		boot.bootstrap(task, false);
		this.seq2SeqLSTM = boot.getSeq2SeqLSTM();
		System.out.println("Finished with LSTM bootstrap..");
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

	int trainRound = 0;
	@Override
	public void trainLocal() throws Exception {		
		System.out.println("Train locally with in Q mode ? "+seq2SeqLSTM.getCfg().isTestQ());
		LSTMConfigurator cfg = seq2SeqLSTM.getCfg().getQlSTMConfigurator();
		LSTM clstm = seq2SeqLSTM.getQlstm();
		if (!seq2SeqLSTM.getCfg().isTestQ()) {
			cfg = seq2SeqLSTM.getCfg().getAlSTMConfigurator();
			clstm = seq2SeqLSTM.getAlstm();
		}
		trainRound ++;
		int taskSize = (task.getEnd() - task.getStart() + 1);	
		cfg.setMBSize(task.getMbatch());
		
		seq2SeqLSTM.trainModel(false, boot.getQsi(), boot.getAsi(), boot.getNna());
		
		int loop = getLoopNum(taskSize, task.getMbatch());
		
		if (trainRound == loop) {
//			cfg.setMBSize(taskSize - task.getMbatch() * (loop - 1));
			trainRound = 0;
			clstm.finish1Cycle();
		} else {
			
		}
		System.out.println("Complete with batch size "+task.getMbatch());
	}

	@Override
	public Error getError() {
//		System.out.println("prepare the Error");
		Error err = new Error();
		if (seq2SeqLSTM.getCfg().isTestQ()) {
			err.setReady(seq2SeqLSTM.getQlstm().isFinish1Cycle());
			err.setCnt(seq2SeqLSTM.getCfg().getQlSTMConfigurator().getMiniBatchSize());
			err.setErr(seq2SeqLSTM.getQlstm().getError());
		} else {
			err.setReady(seq2SeqLSTM.getAlstm().isFinish1Cycle());
			err.setCnt(seq2SeqLSTM.getCfg().getAlSTMConfigurator().getMiniBatchSize());
			err.setErr(seq2SeqLSTM.getAlstm().getError());
		}		
		System.out.println("Error is ready");
		return err;
	}

	LSTMWwUpdater WwUpdater = new LSTMWwUpdater();
	LSTMWwUpdater deltaWwUpdater = new LSTMWwUpdater(false, false);
	boolean testQ = true;
	private Seq2SeqLSTMConfigurator cfgFromSrv;
	
	LSTMCfgCleaner lSTMCfgCleaner = new LSTMCfgCleaner();
	@Override
	public void setSubject(Object obj) {
		cfgFromSrv = (Seq2SeqLSTMConfigurator) obj;
		testQ = cfgFromSrv.isTestQ();
		System.out.println("Subject is from server, " +
				"in Q mode? "+ testQ+", "+cfgFromSrv.getLoop());
		seq2SeqLSTM.getCfg().setTestQ(testQ);
//		testWw(cfgFromSrv.getQlSTMConfigurator());
		
		lSTMCfgCleaner.clean(seq2SeqLSTM.getCfg());
		seq2SeqLSTM.setTrainCfg(cfgFromSrv);
		lSTMCfgCleaner.gbClean();
//		testWw(cfgFromSrv.getQlSTMConfigurator());
//		lSTMWwUpdater.updatewWs(cfgFromSrv.getQlSTMConfigurator()
//				, seq2SeqLSTM.getCfg().getQlSTMConfigurator());
//		lSTMWwUpdater.updatewWs(cfgFromSrv.getAlSTMConfigurator()
//				, seq2SeqLSTM.getCfg().getAlSTMConfigurator());
//		System.out.println("Already fresh Ww from server" );
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
		deltaWwUpdater.updatewWs(seq2SeqLSTM.getCfg().getQlSTMConfigurator(), 
				this.cfgFromSrv.getQlSTMConfigurator());
		deltaWwUpdater.updatewWs(seq2SeqLSTM.getCfg().getAlSTMConfigurator(), 
				this.cfgFromSrv.getAlSTMConfigurator());
		System.out.println("Get the local cfg ready" );
		return cfgFromSrv;
	}

}
