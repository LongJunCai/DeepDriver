package deepDriver.dl.aml.lstm.distribution;

import java.util.List;


import deepDriver.dl.aml.distribution.Error;
import deepDriver.dl.aml.distribution.Master;
import deepDriver.dl.aml.lstm.ICxtConsumer;
import deepDriver.dl.aml.lstm.LSTMCfgCleaner;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMDataSet;
import deepDriver.dl.aml.lstm.LSTMDeltaWwUpdater;
import deepDriver.dl.aml.lstm.LSTMWwFresher;
import deepDriver.dl.aml.lstm.LSTMWwUpdater;
import deepDriver.dl.aml.lstm.Seq2SeqLSTM;
import deepDriver.dl.aml.lstm.Seq2SeqLSTMConfigurator;
import deepDriver.dl.aml.string.Dictionary;

public class Seq2SeqMaster extends Master {
	Dictionary dic;
	Seq2SeqLSTM srvQ2QLSTM;	
	public Seq2SeqMaster() {
		
	}
	Seq2SeqLSTMBoostrapper boot;
	public Seq2SeqMaster(int clientsNum, Seq2SeqLSTMBoostrapper boot) throws Exception {
		super();
		this.boot = boot;
		boot.bootstrap(null, true);
		this.dic = boot.getDic();
		this.srvQ2QLSTM = boot.getSeq2SeqLSTM();
		this.clientsNum = clientsNum;
		taskNum = dic.getLineNum();
		srvQ2QLSTM.getCfg().setTestQ(true);
	}

	int taskNum = 4002;
	int clientsNum = 4;

	@Override
	public int getClientsNum() {
		return clientsNum;
	}
	SimpleTask [] tasks;
	int batchSize = 16;
	int tq = 0;
	@Override
	public Object[] splitTasks() {
//		System.out.println("Prepare to assign task to clients "+clientsNum);
		if (tasks == null) {
			System.out.println("There are "+taskNum +" tasks, and handled by "+clientsNum+" clients");
		}		
		tasks = new SimpleTask[clientsNum];
		tq = taskNum / clientsNum;
//		batchSize = tq;
		int left = taskNum - tq * clientsNum;
		
		for (int i = 0; i < tasks.length; i++) {
			int l = 0;
			if (left >= i+1) {
				l = 1;
			}
			if (i == 0) {
				tasks[i] = new SimpleTask(1, tq + l, batchSize);
			} else {
				tasks[i] = new SimpleTask(
						tasks[i - 1].getEnd()+ 1, tasks[i - 1].getEnd() + tq + l, batchSize);
			}			
		}
//		System.out.println("done ."+clientsNum);
		this.srvQ2QLSTM.getCfg().getQlSTMConfigurator().setMBSize(batchSize);
		this.srvQ2QLSTM.getCfg().getAlSTMConfigurator().setMBSize(batchSize);		
		return tasks;
	}

	private Seq2SeqLSTMConfigurator cfgFromClient;
	LSTMWwUpdater wWUpdater = new LSTMWwUpdater(false, true);
	LSTMWwUpdater deltaWwUpdater = new LSTMWwUpdater(false, false);
	@Override
	public Object getDistributeSubject() {
		System.out.println("Prepare srv cfg to clients"+srvQ2QLSTM.getCfg().getLoop());
		
		if (cfgFromClient == null) {
			cfgFromClient = srvQ2QLSTM.getCfg();
		} else {
			cfgFromClient.setTestQ(srvQ2QLSTM.getCfg().isTestQ());
			cfgFromClient.setLoop(srvQ2QLSTM.getCfg().getLoop());
			wWUpdater.updatewWs(srvQ2QLSTM.getCfg().getQlSTMConfigurator(), 
					cfgFromClient.getQlSTMConfigurator());
			wWUpdater.updatewWs(srvQ2QLSTM.getCfg().getAlSTMConfigurator(), 
					cfgFromClient.getAlSTMConfigurator());
			
			deltaWwUpdater.updatewWs(srvQ2QLSTM.getCfg().getQlSTMConfigurator(), 
					cfgFromClient.getQlSTMConfigurator());
			deltaWwUpdater.updatewWs(srvQ2QLSTM.getCfg().getAlSTMConfigurator(), 
					cfgFromClient.getAlSTMConfigurator());
		}/*****/
		System.out.println("Ready for srv cfg to clients"+srvQ2QLSTM.getCfg().getLoop());
		return cfgFromClient;
	} 
	
	public void testOnMaster() throws Exception {
		if (!done) {
//			System.out.println("THE TRAINING IS NOT DONE, DO NOT TEST FOR NOW");
			return;
		}
		srvQ2QLSTM.getCfg().getQlSTMConfigurator().setAccuracy(-1);
		srvQ2QLSTM.getCfg().getAlSTMConfigurator().setAccuracy(-1);
		srvQ2QLSTM.trainModel(boot.getQsi(), boot.getAsi(), boot.getNna());
		if (srvQ2QLSTM.getCfg().isTestQ()) {
			System.out.println("Test in Q mode");
			testOnMasterWithQ();
		} else {
			System.out.println("Test in A mode");
			testOnMasterWithA();
		}
	}
	
	public void testOnMasterWithQ() throws Exception {
		LSTMConfigurator qcfg = this.srvQ2QLSTM.getCfg().getQlSTMConfigurator();
		ICxtConsumer cxtConsumer = qcfg.
				getCxtConsumer();	
		qcfg.setPreCxtProvider(null);
		qcfg.setCxtConsumer(null);
		String start = "他";
		LSTMDataSet tds = dic.encodeSample(start, 5);
		System.out.println("Start testing..."+start);
		double [][][] ts = srvQ2QLSTM.getQlstm().testModel(tds, 10);
		String ss = dic.decoded(ts);
		System.out.println(ss);
		qcfg.setCxtConsumer(cxtConsumer);
	}
	
	public void testOnMasterWithA() throws Exception {
		String start = "现在我觉得有些孤单";
		System.out.println("Start training...");
//		String start = "我";
		LSTMDataSet qds = dic.encodeSample(start, start.length());
		LSTMDataSet ads = dic.encodeSample("X", 5);
		System.out.println("Start testing..."+start);
		List<double[][][]> ts = srvQ2QLSTM.testModel(qds, ads, 20, 40, 
				ads.getSamples()[0][0]);
		for (int i = 0; i < ts.size(); i++) {
			String ss = dic.decoded(ts.get(i));
			System.out.print(i+ss);
		}
		srvQ2QLSTM.swith2TrainContextLvger();
	}
	
	LSTMDeltaWwUpdater batchDeltaWwUpdater = new LSTMDeltaWwUpdater();
//	LSTMDeltaWwFromWwUpdater batchDeltaWwUpdater = new LSTMDeltaWwFromWwUpdater();
	
	LSTMWwFresher lSTMWwFresher = new LSTMWwFresher();
	LSTMWwUpdater wWchecker = new LSTMWwUpdater(true, false);
	
	LSTMCfgCleaner lSTMCfgCleaner = new LSTMCfgCleaner();

	@Override
	public void mergeSubject(Object[] objs) {
		System.out.println("Prepare to merge Ww from clients");
		Seq2SeqLSTMConfigurator[] seq2seqCfgs = new Seq2SeqLSTMConfigurator[objs.length];
		LSTMConfigurator [] qcfgs = new LSTMConfigurator[seq2seqCfgs.length];
		LSTMConfigurator [] acfgs = new LSTMConfigurator[seq2seqCfgs.length];
		for (int i = 0; i < acfgs.length; i++) {
			seq2seqCfgs[i] = (Seq2SeqLSTMConfigurator) objs[i];
			qcfgs[i] = seq2seqCfgs[i].getQlSTMConfigurator();
			acfgs[i] = seq2seqCfgs[i].getAlSTMConfigurator();
		}
		srvQ2QLSTM.getCfg().setLoop(srvQ2QLSTM.getCfg().getLoop() + 1);
//		testWw(qcfgs[0]);
		LSTMConfigurator qcfg = this.srvQ2QLSTM.getCfg().
				getQlSTMConfigurator();
		batchDeltaWwUpdater.mergeDeltawWs(qcfgs, qcfg, qcfg.getLearningRate(),
				qcfg.getM());
		LSTMConfigurator acfg = this.srvQ2QLSTM.getCfg().
				getQlSTMConfigurator();
		batchDeltaWwUpdater.mergeDeltawWs(acfgs, acfg, acfg.getLearningRate(),
				acfg.getM());
		lSTMWwFresher.freshwWs(this.srvQ2QLSTM.getCfg().
				getQlSTMConfigurator());
		lSTMWwFresher.freshwWs(this.srvQ2QLSTM.getCfg().
				getAlSTMConfigurator());		
		System.out.println("Done with Ww updates");
//		testWw(qcfgs[0]);
		
		if (cfgFromClient != srvQ2QLSTM.getCfg()) {
			lSTMCfgCleaner.clean(cfgFromClient);
		}		
		cfgFromClient = seq2seqCfgs[0];
		qcfgs[0] = null;
		acfgs[0] = null;
		for (int i = 1; i < seq2seqCfgs.length; i++) {
			lSTMCfgCleaner.clean(seq2seqCfgs[i]);
			seq2seqCfgs[i] = null;
			qcfgs[i] = null;
			acfgs[i] = null;
		}
		lSTMCfgCleaner.gbClean();
		
	}
	
	public void testWw(LSTMConfigurator fcfg) {
		System.out.println("Prepare to check.");
		wWchecker.updatewWs(fcfg, srvQ2QLSTM.getCfg().
				getQlSTMConfigurator());
		System.out.println("done with check.");
	}
	
	public static void testSplitTasks() {
		Seq2SeqMaster seq2SeqMaster = new Seq2SeqMaster();
		seq2SeqMaster.splitTasks();		
	}

	@Override
	public double caculateErrorLastTime(Object[] objs) {
		double err = 0;
		double cnt = 0;	
		Error ce1 = (Error)objs[0];
		if (!ce1.isReady()) {
			return 0;
		}
		for (int i = 0; i < objs.length; i++) {
			Error ce = (Error)objs[i];
			err = err + ce.getErr();
			cnt = cnt + 1;//+ ce.getCnt();
		}
		err = err/cnt;
		if (srvQ2QLSTM.getCfg().isTestQ()) {
			if (err <= srvQ2QLSTM.getCfg().getQlSTMConfigurator().getAccuracy()) {
				srvQ2QLSTM.getCfg().setTestQ(false);
				System.out.println("Switch to A mode");
				cfgFromClient = null;
			}
			System.out.println(cnt+" is tested Q with error "+ err);
		} else {
			if (err <= srvQ2QLSTM.getCfg().getAlSTMConfigurator().getAccuracy()) {
				this.done = true;
				System.out.println("Complete training");
			}
			System.out.println(cnt+" is tested A with error "+ err);
		}		
		return err;
	}
}
