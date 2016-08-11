package deepDriver.dl.aml.lstm.distribution;

import java.util.List;


import deepDriver.dl.aml.lstm.ICxtConsumer;
import deepDriver.dl.aml.lstm.ITest;
import deepDriver.dl.aml.lstm.LSTM;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMDataSet;
import deepDriver.dl.aml.lstm.NeuroNetworkArchitecture;
import deepDriver.dl.aml.lstm.Seq2SeqLSTM;
import deepDriver.dl.aml.string.ANFixedStreamImpV2;
import deepDriver.dl.aml.string.Dictionary;
import deepDriver.dl.aml.string.NFixedStreamImpV2;

public class Seq2SeqLSTMSetup implements Seq2SeqLSTMBoostrapper {
	
	final int t = 5;
	Dictionary dic = new Dictionary();	
	NeuroNetworkArchitecture nna = new NeuroNetworkArchitecture();
	NFixedStreamImpV2 qsi;
	ANFixedStreamImpV2 asi;
	Seq2SeqLSTM seq2SeqLSTM;
	
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

	public void bootstrap(SimpleTask task, boolean isServer) throws Exception {
		dic.loadDicFromFile("D:\\6.workspace\\ANN\\wangfeng_V2.txt");
//		dic.loadDicFromFile("D:\\6.workspace\\ANN\\wangfeng.txt");
		final LSTMConfigurator qcfg = new LSTMConfigurator();
		qcfg.setBinaryLearning(true);
		qcfg.setLoopNum(1);
		qcfg.setAccuracy(30);
//		if (isServer) {
//			qcfg.setAccuracy(-1);
//		}
		qcfg.setMaxTimePeriod(40);
		qcfg.setLearningRate(0.01);
		qcfg.setM(0.5);
		qcfg.setMBSize(1);
		if (task != null) {
			qcfg.setMBSize(task.getMbatch());
		}		
		qcfg.setDropOut(0);
		qcfg.setEnableUseCellAa(true);
		
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
		acfg.setAccuracy(4);
//		if (isServer) {
//			qcfg.setAccuracy(-1);
//		}
		acfg.setMaxTimePeriod(40);
		acfg.setLearningRate(0.01);
		acfg.setM(0.5);
		acfg.setMBSize(1);
		if (task != null) {
			acfg.setMBSize(task.getMbatch());
		}		
		acfg.setDropOut(0);
		acfg.setEnableUseCellAa(true);
		final LSTM alstm = new LSTM(acfg);
		if (task == null) {
			qsi = new NFixedStreamImpV2(dic, t);
			asi = new ANFixedStreamImpV2(dic, t, 1);
		} else {
			qsi = new NFixedStreamImpV2(dic, t, task.getStart(), task.getEnd());
			asi = new ANFixedStreamImpV2(dic, t, 1 + task.getStart(), task.getEnd());
		}
		
//		LSTMDataSet lds = dic.encodeSamples(20);
		nna.setNnArch(new int [] {110});
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
	public void prepareData(boolean isServer) throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setDic(Dictionary dic) {
		this.dic = dic;
	}

}
