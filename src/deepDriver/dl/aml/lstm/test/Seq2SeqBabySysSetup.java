package deepDriver.dl.aml.lstm.test;

import java.io.File;
import java.util.List;


import deepDriver.dl.aml.distribution.DistributionEnvCfg;
import deepDriver.dl.aml.lstm.ICxtConsumer;
import deepDriver.dl.aml.lstm.ITest;
import deepDriver.dl.aml.lstm.LSTM;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMDataSet;
import deepDriver.dl.aml.lstm.NeuroNetworkArchitecture;
import deepDriver.dl.aml.lstm.Seq2SeqLSTM;
import deepDriver.dl.aml.lstm.distribution.Seq2SeqLSTMBoostrapper;
import deepDriver.dl.aml.lstm.distribution.SimpleTask;
import deepDriver.dl.aml.string.ANFixedStreamImpV2;
import deepDriver.dl.aml.string.Dictionary;
import deepDriver.dl.aml.string.NFixedStreamImpV2;
import deepDriver.dl.aml.string.ThinRandomANFixedStreamImpV2;
import deepDriver.dl.aml.string.ThinRandomQNFixedStreamImpV2;

public class Seq2SeqBabySysSetup implements Seq2SeqLSTMBoostrapper {
	
	final int t = 5;
	Dictionary dic = new Dictionary();	
	NeuroNetworkArchitecture nna = new NeuroNetworkArchitecture();
	NFixedStreamImpV2 qsi;
	ANFixedStreamImpV2 asi;
	Seq2SeqLSTM seq2SeqLSTM;
	
	boolean useFullDic = false;
	
	public boolean isUseFullDic() {
		return useFullDic;
	}

	public void setUseFullDic(boolean useFullDic) {
		this.useFullDic = useFullDic;
	}

	public NeuroNetworkArchitecture getNna() {
		return nna;
	}

	public void setNna(NeuroNetworkArchitecture nna) {
		this.nna = nna;
	}

	public NFixedStreamImpV2 getQsi() {
		return qsi;
	}

	public void setQsi(NFixedStreamImpV2 qsi) {
		this.qsi = qsi;
	}

	public ANFixedStreamImpV2 getAsi() {
		return asi;
	}

	public void setAsi(ANFixedStreamImpV2 asi) {
		this.asi = asi;
	}

	public Seq2SeqLSTM getSeq2SeqLSTM() {
		return seq2SeqLSTM;
	}

	public void setSeq2SeqLSTM(Seq2SeqLSTM seq2SeqLSTM) {
		this.seq2SeqLSTM = seq2SeqLSTM;
	}

	public Dictionary getDic() {
		return dic;
	}
	
	public static String KEY_FS_ROOT = "KEY_FS_ROOT"; 
	public static String KEY_TEST_FILE = "KEY_TEST_FILE"; 
	
	int mbSize = 80;
	
	String fsRoot = "D:\\6.workspace\\ANN\\";
	String testFile = "wangfeng.txt";
//	String testFile = "wangfeng_V2.txt";
	
	public void prepareData(boolean isServer) throws Exception {
		String root = DistributionEnvCfg.getCfg().getString(KEY_FS_ROOT);
		if (root != null) {
			fsRoot = root;
		} 
		
		String tf = DistributionEnvCfg.getCfg().getString(KEY_TEST_FILE);
		if (tf != null) {
			testFile = tf;
		} 
		if (isServer) {
			dic.setDebug(false);
		}
//		dic.setUseParser(true);//talk
		dic.setUseParser(true);//weibo
		//
		if (!useFullDic) {
			dic.setUsageSortedNum(1000);
		}	
		//
		String sf = System.getProperty("user.dir");
		File f = new File(sf, "data/"+testFile);
		if (f.exists()) {
			dic.loadDicFromFile(f.getAbsolutePath());
		} else {
			dic.loadDicFromFile(fsRoot+testFile);
		}
		//		dic.loadDicFromFile(fsRoot+"wangfeng_V2.txt");
		prepared = true;
	}
	
	boolean prepared = false;
	int threadsNum = 4;	
	
	public int getThreadsNum() {
		return threadsNum;
	}

	public void setThreadsNum(int threadsNum) {
		this.threadsNum = threadsNum;
	}

	public void bootstrap(SimpleTask task, boolean isServer) throws Exception {
		if (!prepared) {
			prepareData(isServer);
		}
		
		final LSTMConfigurator qcfg = new LSTMConfigurator();
		qcfg.setBinaryLearning(true);
		qcfg.setLoopNum(1);
		qcfg.setAccuracy(40);
//		if (isServer) {
//			qcfg.setAccuracy(-1);
//		}
		qcfg.setMaxTimePeriod(40);
		qcfg.setLearningRate(0.001);//talk
//		qcfg.setLearningRate(0.01);//? 
//		qcfg.setLearningRate(1.9753086419753085E-4);
		qcfg.setM(0.6);
		qcfg.setMBSize(mbSize);
		if (task != null) {
			qcfg.setMBSize(task.getMbatch());
		}		
		qcfg.setDropOut(0);
		qcfg.setEnableUseCellAa(true);
		qcfg.setBatchSize4DeltaWw(true);//?
		qcfg.setUseRmsProp(false);
//		qcfg.setInteractiveUpdate(true);
		qcfg.setUseThinData(true);
		qcfg.setUseBias(true);
		
		qcfg.setThreadsNum(threadsNum);//?
		
		final LSTM qlstm = new LSTM(qcfg);
		if (isServer) {
			qlstm.setPhaseTest(new ITest() {
			public void test() throws Exception { 
				ICxtConsumer cxtConsumer = qcfg.getCxtConsumer();	
				qcfg.setPreCxtProvider(null);
				qcfg.setCxtConsumer(null);
				String start = "他";
				LSTMDataSet tds = dic.encodeSample(start, t);
				System.out.println("Start testing..."+start);
				double [][][] ts = qlstm.testModel(tds, 10);
				String ss = dic.decoded(ts);
				System.out.println(ss);
				qcfg.setCxtConsumer(cxtConsumer);
			}
			});
		}
		
		
		LSTMConfigurator acfg = new LSTMConfigurator();
		acfg.setBinaryLearning(true);
		acfg.setLoopNum(1);
		acfg.setAccuracy(40);
//		if (isServer) {
//			qcfg.setAccuracy(-1);
//		}
		acfg.setMaxTimePeriod(41);
		acfg.setLearningRate(0.01);
//		acfg.setLearningRate(1.9753086419753085E-4);
		acfg.setM(0.6);
		acfg.setMBSize(mbSize);
		if (task != null) {
			acfg.setMBSize(task.getMbatch());
		}		
		acfg.setDropOut(0);
		acfg.setEnableUseCellAa(true);
		acfg.setBatchSize4DeltaWw(true);
		acfg.setUseRmsProp(false);
		acfg.setUseThinData(true);
		acfg.setUseBias(true);
		acfg.setThreadsNum(threadsNum);//?
		
		final LSTM alstm = new LSTM(acfg);
		long l = System.currentTimeMillis();
		if (task == null) {
			qsi = new ThinRandomQNFixedStreamImpV2(dic, t, l);
			asi = new ThinRandomANFixedStreamImpV2(dic, t, l);
		} else {
			qsi = new ThinRandomQNFixedStreamImpV2(dic, t, task.getStart(), task.getEnd(), l);
			asi = new ThinRandomANFixedStreamImpV2(dic, t, task.getStart(), task.getEnd(), l);
		}
		
//		LSTMDataSet lds = dic.encodeSamples(20);
		nna.setNnArch(new int [] {128, 128});
		nna.setCostFunction(LSTMConfigurator.SOFT_MAX);
		qcfg.buildArchitecture(qsi, nna);
		acfg.buildArchitecture(asi, nna);
		seq2SeqLSTM = new Seq2SeqLSTM(qlstm, alstm);
		if (isServer) {	
			alstm.setPhaseTest(new ITest() {			
				@Override
				public void test() throws Exception {
					String start = "现在我觉得有些孤单";
					System.out.println("Start training...");
//					String start = "我";
					LSTMDataSet qds = dic.encodeSample(start, start.length());
					LSTMDataSet ads = dic.encodeSample("X", t);
					System.out.println("Start testing..."+start);
					List<double[][][]> ts = seq2SeqLSTM.testModel(qds, ads, 20, 40, 
						ads.getSamples()[0][0]);
					for (int i = 0; i < ts.size(); i++) {
						String ss = dic.decoded(ts.get(i));
						System.out.print(i+ss);
					}
					seq2SeqLSTM.swith2TrainContextLvger();
				}
			});
		}
//		seq2SeqLSTM.trainModel(qsi, asi, nna); 	
	}

	@Override
	public void setDic(Dictionary dic) {
		this.dic = dic;
		prepared = true;
	}

}
