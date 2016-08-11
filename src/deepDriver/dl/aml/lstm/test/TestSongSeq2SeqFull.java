package deepDriver.dl.aml.lstm.test;

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

public class TestSongSeq2SeqFull {
	
	public static void main(String[] args) throws Exception {
		final Dictionary dic = new Dictionary();
//		dic.loadDicFromFile("D:\\6.workspace\\ANN\\wangfeng_V2.txt");
		dic.loadDicFromFile("D:\\6.workspace\\ANN\\wangfeng.txt");

		final int t = 5;
		final LSTMConfigurator qcfg = new LSTMConfigurator();
		qcfg.setBinaryLearning(true);
		qcfg.setLoopNum(30);
		qcfg.setAccuracy(6);
		qcfg.setMaxTimePeriod(40);
		qcfg.setLearningRate(0.01);
		qcfg.setM(0.5);
		qcfg.setMBSize(128);
		qcfg.setDropOut(0);
		qcfg.setEnableUseCellAa(true);
		
		final LSTM qlstm = new LSTM(qcfg);
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
		
		LSTMConfigurator acfg = new LSTMConfigurator();
		acfg.setBinaryLearning(true);
		acfg.setLoopNum(10);
		acfg.setAccuracy(6);
		acfg.setMaxTimePeriod(40);
		acfg.setLearningRate(0.01);
		acfg.setM(0.5);
		acfg.setMBSize(128);
		acfg.setDropOut(0);
		acfg.setEnableUseCellAa(true);
		final LSTM alstm = new LSTM(acfg);
		NFixedStreamImpV2 qsi = new NFixedStreamImpV2(dic, t);
		ANFixedStreamImpV2 asi = new ANFixedStreamImpV2(dic, t, 1);
		
//		LSTMDataSet lds = dic.encodeSamples(20);
		NeuroNetworkArchitecture nna = new NeuroNetworkArchitecture();
		nna.setNnArch(new int [] {110});
		nna.setCostFunction(LSTMConfigurator.SOFT_MAX);
		qcfg.buildArchitecture(qsi, nna);
		acfg.buildArchitecture(asi, nna);
		final Seq2SeqLSTM seq2SeqLSTM = new Seq2SeqLSTM(qlstm, alstm);
			
		alstm.setPhaseTest(new ITest() {			
			@Override
			public void test() throws Exception {
				String start = "现在我觉得有些孤单";
				System.out.println("Start training...");
//				String start = "我";
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
		seq2SeqLSTM.trainModel(qsi, asi, nna); 
		
	}

}
