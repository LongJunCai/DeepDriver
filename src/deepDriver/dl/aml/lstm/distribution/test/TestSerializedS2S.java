package deepDriver.dl.aml.lstm.distribution.test;

import java.util.List;


import deepDriver.dl.aml.distribution.DistributionEnvCfg;
import deepDriver.dl.aml.distribution.Fs;
import deepDriver.dl.aml.lstm.ICxtConsumer;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMDataSet;
import deepDriver.dl.aml.lstm.LSTMWwUpdater;
import deepDriver.dl.aml.lstm.Seq2SeqLSTM;
import deepDriver.dl.aml.lstm.Seq2SeqLSTMConfigurator;
import deepDriver.dl.aml.lstm.distribution.Seq2SeqLSTMBoostrapper;
import deepDriver.dl.aml.string.Dictionary;

public class TestSerializedS2S {
	Dictionary dic;
	Seq2SeqLSTM srvQ2QLSTM;	 
	Seq2SeqLSTMBoostrapper boot;
	LSTMWwUpdater wWUpdater = new LSTMWwUpdater(false, true);
	public void load(Seq2SeqLSTMBoostrapper boot, String file) throws Exception {
		boot.bootstrap(null, true);
		this.dic = boot.getDic();
		this.srvQ2QLSTM = boot.getSeq2SeqLSTM();
		Seq2SeqLSTMConfigurator cfg = (Seq2SeqLSTMConfigurator) Fs.readObjFromFile(file);
		System.out.println(cfg.getQlSTMConfigurator().getLearningRate());
		System.out.println(cfg.getAlSTMConfigurator().getLearningRate());
		checker.updatewWs(cfg.getQlSTMConfigurator(), srvQ2QLSTM.getCfg().getQlSTMConfigurator());
		wWUpdater.updatewWs(cfg.getQlSTMConfigurator(), srvQ2QLSTM.getCfg().getQlSTMConfigurator());
		wWUpdater.updatewWs(cfg.getAlSTMConfigurator(), 
				srvQ2QLSTM.getCfg().getAlSTMConfigurator());
		System.out.println("Complete the merging..");
		checker.updatewWs(cfg.getQlSTMConfigurator(), srvQ2QLSTM.getCfg().getQlSTMConfigurator());

		System.out.println("q l="+srvQ2QLSTM.getCfg().getQlSTMConfigurator().getLearningRate()
				+ ", m="+srvQ2QLSTM.getCfg().getQlSTMConfigurator().getM());
		System.out.println("a l="+srvQ2QLSTM.getCfg().getAlSTMConfigurator().getLearningRate()
				+ ", m="+srvQ2QLSTM.getCfg().getAlSTMConfigurator().getM());
		this.srvQ2QLSTM.rebuildCfg(srvQ2QLSTM.getCfg());
//		srvQ2QLSTM.getCfg().getQlSTMConfigurator().setUseRandomResult(true);
//		srvQ2QLSTM.getCfg().getAlSTMConfigurator().setUseRandomResult(true);
	}
	
	LSTMWwUpdater checker = new LSTMWwUpdater(true, true);
	public void check(String s1, String s2) throws Exception {
		Seq2SeqLSTMConfigurator cfg1 = (Seq2SeqLSTMConfigurator) Fs.readObjFromFile(s1);
		Seq2SeqLSTMConfigurator cfg2 = (Seq2SeqLSTMConfigurator) Fs.readObjFromFile(s2);
		checker.updatewWs(cfg1.getQlSTMConfigurator(), cfg2.getQlSTMConfigurator());
	}
	
	
	public void testOnMasterWithQ() throws Exception {
		LSTMConfigurator qcfg = this.srvQ2QLSTM.getCfg().getQlSTMConfigurator();
		ICxtConsumer cxtConsumer = qcfg.
				getCxtConsumer();	
		qcfg.setPreCxtProvider(null);
		qcfg.setCxtConsumer(null);
		String start = "他觉得";
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
	
	public static void main(String[] args) throws Exception {
		TestSerializedS2S testSerializedS2S = new TestSerializedS2S();
		String root = "D:\\6.workspace\\ANN\\lstm\\";
		Seq2SeqLSTMSetup s2s = new Seq2SeqLSTMSetup();
//		s2s.setUseFullDic(true);
		DistributionEnvCfg.getCfg(). set(Seq2SeqLSTMSetup.KEY_FS_ROOT, root);
		DistributionEnvCfg.getCfg(). set(Seq2SeqLSTMSetup.KEY_TEST_FILE, "talk2016.txt");
//		DistributionEnvCfg.getCfg(). set(P2PServer.KEY_SRV_HOST, args[2]);
		testSerializedS2S.load(s2s, 
//				root+"seq2seqCfg_1452821497781.m1452821497781");
//		root+"seq2seqCfg_1452562670182.m1452562670416");
//		root+"seq2seqCfg_1452994429866.m1452994429866");
		root+"a_lstm_1466991386152_0.m");
		testSerializedS2S.testOnMasterWithQ();
//		testSerializedS2S.testOnMasterWithA();
		
//		testSerializedS2S.check( 
//				root+  "seq2seqCfg_1452821497781.m1452821497781",
//				root + "seq2seqCfg_1452562670182.m1452562670416" );
		
	}
	
	

}
