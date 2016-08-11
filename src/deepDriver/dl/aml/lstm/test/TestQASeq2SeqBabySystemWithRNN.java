package deepDriver.dl.aml.lstm.test;

import java.util.List;


import deepDriver.dl.aml.distribution.Fs;
import deepDriver.dl.aml.lrate.StepReductionLR;
import deepDriver.dl.aml.lstm.ICxtConsumer;
import deepDriver.dl.aml.lstm.ITest;
import deepDriver.dl.aml.lstm.LSTM;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMDataSet;
import deepDriver.dl.aml.lstm.LSTMWwUpdater;
import deepDriver.dl.aml.lstm.NeuroNetworkArchitecture;
import deepDriver.dl.aml.lstm.Seq2SeqLSTM;
import deepDriver.dl.aml.string.ANFixedStreamImpV2;
import deepDriver.dl.aml.string.Dictionary;
import deepDriver.dl.aml.string.NFixedStreamImpV2;
import deepDriver.dl.aml.string.ThinRandomANFixedStreamImpV2;
import deepDriver.dl.aml.string.ThinRandomQNFixedStreamImpV2;

public class TestQASeq2SeqBabySystemWithRNN {
	
	static int qFile = 1;
	
	public static void main(String[] args) throws Exception {
		final Dictionary dic = new Dictionary();
//		dic.loadDicFromFile("D:\\6.workspace\\ANN\\wangfeng_V2.txt");
		dic.setUsageSortedNum(1000);
//		dic.setDebug(true);
		int maxLength = 40;
		dic.setMaxLength(maxLength);
		dic.setUseParser(true);
//		dic.loadDicFromFile("D:\\6.workspace\\ANN\\lstm\\ideaInf");
		dic.loadDicFromFile("D:\\6.workspace\\ANN\\lstm\\talk2016.txt");
		
		String sqFile = null;//aa    D:\\workspace\\ANN\\data\\q_lstm_1464485024333_0.m
		if (args.length >= qFile) {
			sqFile = args[qFile];
		}

		final int t = 5;
		final LSTMConfigurator qcfg = new LSTMConfigurator();
		qcfg.setBinaryLearning(true);
		qcfg.setLoopNum(30);
		qcfg.setAccuracy(10);
		qcfg.setMaxTimePeriod(maxLength);
		qcfg.setLearningRate(0.1);
		qcfg.setM(0.2);
		qcfg.setMBSize(1);
		qcfg.setDropOut(0);
		qcfg.setEnableUseCellAa(true);
		qcfg.setBatchSize4DeltaWw(true);
		qcfg.setUseRmsProp(false);
		qcfg.setUseThinData(true);
		qcfg.setUseBias(true);
		StepReductionLR slr = new StepReductionLR();
		slr.setStepsCnt(60000);
		slr.setReductionRate(0.5);
		slr.setMinLr(0.001);
		qcfg.setLr(slr);
		
		StepReductionLR srm = new StepReductionLR();
		srm.setStepsCnt(30000);
		srm.setReductionRate(0.5);
		srm.setMinLr(0.01);
		qcfg.setSrm(srm);
//		qcfg.setDebug(true);
//		qcfg.setForceComplete(true);
		
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
		acfg.setMaxTimePeriod(maxLength + 1);
		acfg.setLearningRate(0.1);
		acfg.setM(0.6);
		acfg.setMBSize(1);
		acfg.setDropOut(0);
		acfg.setEnableUseCellAa(true);
		acfg.setUseThinData(true);
		acfg.setUseBias(true);
		final LSTM alstm = new LSTM(acfg);
		
//		NFixedStreamImpV2 qsi = new RandomQNFixedStreamImpV2(dic, t);
		long l = System.currentTimeMillis();
		NFixedStreamImpV2 qsi = new ThinRandomQNFixedStreamImpV2(dic, t, l);
		ANFixedStreamImpV2 asi = new ThinRandomANFixedStreamImpV2(dic, t, l);
		
//		LSTMDataSet lds = dic.encodeSamples(20);
		NeuroNetworkArchitecture nna = new NeuroNetworkArchitecture();
		nna.setHiddenType(NeuroNetworkArchitecture.HiddenRNN);
//		nna.setUseProjectionLayer(true);
//		nna.setNnArch(new int [] {100, 128, 128});
		nna.setNnArch(new int [] {128, 128});
		nna.setCostFunction(LSTMConfigurator.SOFT_MAX);
		qcfg.buildArchitecture(qsi, nna);
		acfg.buildArchitecture(asi, nna);
		
		if (sqFile != null) {
			System.out.println("Upgrade qCfg from file: "+sqFile);
			LSTMWwUpdater wWUpdater = new LSTMWwUpdater(false, true);
			LSTMWwUpdater deltaWwUpdater = new LSTMWwUpdater(false, false);
			LSTMWwUpdater checker = new LSTMWwUpdater(true, true);
			LSTMConfigurator cfg = (LSTMConfigurator) Fs.readObjFromFile(sqFile);
			checker.updatewWs(cfg, qcfg);
			wWUpdater.updatewWs(cfg, qcfg); 
			deltaWwUpdater.updatewWs(cfg, qcfg); 
			System.out.println("Complete the merging..");
			checker.updatewWs(cfg, qcfg);
		}
//		
		
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
