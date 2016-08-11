package deepDriver.dl.aml.lstm.test;

import java.io.File;
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

public class TestS2S {
	Dictionary dic;
	Seq2SeqLSTM srvQ2QLSTM;	 
	Seq2SeqLSTMBoostrapper boot;
	LSTMWwUpdater wWUpdater = new LSTMWwUpdater(false, true);
	public void load(Seq2SeqBabySysSetup boot, String qfile, String afile, String dicFile) throws Exception {
		File dfile = new File(dicFile);		
		if (dfile.exists()) {
			dic = (Dictionary) Fs.readObjFromFile(dicFile);
			boot.dic = dic;
			boot.prepared = true;
			boot.bootstrap(null, true);
		} else {
			boot.bootstrap(null, true);
			this.dic = boot.getDic();
			Fs.writeObject2File(dicFile, dic);
		}
				
		this.srvQ2QLSTM = boot.getSeq2SeqLSTM();
		
		LSTMConfigurator qcfg = (LSTMConfigurator) Fs.readObjFromFile(qfile);
		LSTMConfigurator acfg = (LSTMConfigurator) Fs.readObjFromFile(afile);
		System.out.println(qcfg.getLearningRate());
		System.out.println(acfg.getLearningRate());
		checker.updatewWs(qcfg, srvQ2QLSTM.getCfg().getQlSTMConfigurator());
		wWUpdater.updatewWs(qcfg, srvQ2QLSTM.getCfg().getQlSTMConfigurator());
		wWUpdater.updatewWs(acfg, 
				srvQ2QLSTM.getCfg().getAlSTMConfigurator());
		System.out.println("Complete the merging..");
		checker.updatewWs(qcfg, srvQ2QLSTM.getCfg().getQlSTMConfigurator());

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
//		qcfg.setUseRandomResult(true);
		String start = "我看视频";//我冲了会员
		LSTMDataSet tds = dic.encodeSample(start, 5, qcfg.isUseThinData());
		System.out.println("Start testing..."+start);
//		srvQ2QLSTM.getCfg().getAlSTMConfigurator().setUseRandomResult(true);
		double [][][] ts = srvQ2QLSTM.getQlstm().testModel(tds, 15);
		String ss = dic.decoded(ts, qcfg.isUseThinData());
		ss = start + ss.substring(start.length() - 1, ss.length());
		System.out.println(ss);
		qcfg.setCxtConsumer(cxtConsumer);
	}
	
	public void testOnMasterWithA() throws Exception {
		LSTMConfigurator qcfg = this.srvQ2QLSTM.getCfg().getQlSTMConfigurator();
		String start = "我看视频一直缓冲为什么";//登录不上去，怎么修
		System.out.println("Start training...");
//		String start = "我";
		LSTMDataSet qds = dic.encodeSample(start, start.length(), qcfg.isUseThinData());
		LSTMDataSet ads = dic.encodeSample(dic.EOS, 1, qcfg.isUseThinData());
		System.out.println("Start testing..."+start);
		srvQ2QLSTM.swith2TrainContextLvger();
		List<double[][][]> ts = srvQ2QLSTM.testModel(qds, ads, 1, 40, 
				ads.getSamples()[0][0]);
		for (int i = 0; i < ts.size(); i++) {
			String ss = dic.decoded(ts.get(i), qcfg.isUseThinData());
			System.out.print(i+ss);
		}
		srvQ2QLSTM.swith2TrainContextLvger();
	}
	
	public void testOnMasterWithA3() throws Exception {
		LSTMConfigurator qcfg = this.srvQ2QLSTM.getCfg().getQlSTMConfigurator();
		String start = "你工号多少，我投诉你";//登录不上去，怎么修//什么是激活码//我看视频一直缓冲为什么
		//我IP地址被封了,请帮我解除//你是猪？
		//为什么无法播放会员电影//我账号密码是什么？//为什么老子付了钱还不是会员
		//是人工吗//您的手机是什么系统的//怎么我的是黄金套餐还要购买才能看电影？
		//改密码，手机号注销了,怎么办//怎么查会员到没到期//什么是订单号,支付凭证
		System.out.println("Start training...");
//		String start = "我";
		LSTMDataSet qds = dic.encodeSample(start, start.length(), qcfg.isUseThinData());
		LSTMDataSet ads = dic.encodeSample(dic.EOS, 1, qcfg.isUseThinData());
		System.out.println("Start testing..."+start);
		srvQ2QLSTM.swith2TrainContextLvger();
		List<double[][][]> ts = srvQ2QLSTM.beamSearch(qds, ads, 10, 40, 
				ads.getSamples()[0][0]);
		for (int i = 0; i < ts.size(); i++) {
			String ss = dic.decoded(ts.get(i), qcfg.isUseThinData(), true);
			System.out.println(i+ss);
		}
		srvQ2QLSTM.swith2TrainContextLvger();
	}
	
	public static void main(String[] args) throws Exception {
		TestS2S testSerializedS2S = new TestS2S();
		String root = "D:\\6.workspace\\ANN\\lstm\\";
		Seq2SeqBabySysSetup s2s = new Seq2SeqBabySysSetup();
//		s2s.setUseFullDic(useFullDic);
//		s2s.setUseFullDic(true);
		DistributionEnvCfg.getCfg(). set(Seq2SeqBabySysSetup.KEY_FS_ROOT, root);
		DistributionEnvCfg.getCfg(). set(Seq2SeqBabySysSetup.KEY_TEST_FILE, "talk2016.txt");
//		DistributionEnvCfg.getCfg(). set(P2PServer.KEY_SRV_HOST, args[2]);
		testSerializedS2S.load(s2s, 
//				root+"seq2seqCfg_1452821497781.m1452821497781");
//		root+"seq2seqCfg_1452562670182.m1452562670416");
//		root+"seq2seqCfg_1452994429866.m1452994429866");
				root+"q_lstm_1466671795460_1.m" , root+"a_lstm_1467890329260_0.m",//a_lstm_1467251362500_1.m//a_lstm_1466991386152_0.m
				root+"dic.m");
		testSerializedS2S.testOnMasterWithQ();
		testSerializedS2S.testOnMasterWithA3();
		
//		testSerializedS2S.check( 
//				root+  "seq2seqCfg_1452821497781.m1452821497781",
//				root + "seq2seqCfg_1452562670182.m1452562670416" );
		
	}
	
	

}
