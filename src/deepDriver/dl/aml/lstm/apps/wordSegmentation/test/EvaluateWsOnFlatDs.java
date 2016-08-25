package deepDriver.dl.aml.lstm.apps.wordSegmentation.test;

import deepDriver.dl.aml.distribution.Fs;
import deepDriver.dl.aml.lrate.StepReductionLR;
import deepDriver.dl.aml.lstm.LSTM;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMWwUpdater;
import deepDriver.dl.aml.lstm.NeuroNetworkArchitecture;
import deepDriver.dl.aml.lstm.apps.wordSegmentation.WordSegSet;
import deepDriver.dl.aml.lstm.apps.wordSegmentation.WordSegSetV2;
import deepDriver.dl.aml.lstm.apps.wordSegmentation.WordSegmentationStream;

public class EvaluateWsOnFlatDs {

	
	static int qFile = 0;
	static int testA = 2;
	
	public static void main(String[] args) throws Exception {
		WordSegSetV2 wss = new WordSegSetV2();
		wss.setVoLoadOnly(true);		
		wss.loadWordSegSet("D:\\6.workspace\\p.NLP\\train.conll");
		 
		wss.setVoLoadOnly(true);
		wss.loadWordSegSet("D:\\6.workspace\\p.NLP\\dev.conll"); 
		
		wss.setVoLoadOnly(false);		
		wss.setLockVo(true);
		
		String flatFile = "msr_corpus";
		if (args.length > 1) {
			flatFile = args[1];
		}
		wss.loadFlatDs("D:\\6.workspace\\p.NLP\\"+flatFile); 

		final int t = 5;
		final LSTMConfigurator qcfg = new LSTMConfigurator();
		qcfg.setBiDirection(true);
		qcfg.setBinaryLearning(true);
		qcfg.setLoopNum(30);
		qcfg.setAccuracy(1);
		qcfg.setMaxTimePeriod(40);
		qcfg.setLearningRate(0.01);//?
		qcfg.setM(0.1);//?
		qcfg.setMBSize(1);
		qcfg.setDropOut(0);
		qcfg.setEnableUseCellAa(true);
		qcfg.setBatchSize4DeltaWw(true);
		qcfg.setUseRmsProp(false);
		qcfg.setUseThinData(true);
		qcfg.setUseBias(true);
		qcfg.setThreadsNum(4);//?
		StepReductionLR slr = new StepReductionLR();
		slr.setStepsCnt(20000);
		slr.setReductionRate(0.5);
		slr.setMinLr(0.001);
		qcfg.setLr(slr);
		
		StepReductionLR srm = new StepReductionLR();
		srm.setStepsCnt(30000);
		srm.setReductionRate(0.5);
		srm.setMinLr(0.01);
		qcfg.setSrm(srm);
		qcfg.setName("Tws");
//		qcfg.setDebug(true);
//		qcfg.setForceComplete(true);
		
		final LSTM qlstm = new LSTM(qcfg);
		
		long l = System.currentTimeMillis();
		WordSegmentationStream qsi = new WordSegmentationStream(wss);
		qsi.setRandom(false);
		
		NeuroNetworkArchitecture nna = new NeuroNetworkArchitecture();

		nna.setNnArch(new int [] {128, 128});
		nna.setCostFunction(LSTMConfigurator.SOFT_MAX);
		qcfg.buildArchitecture(qsi, nna);
		String mfName = "Tws_1471927306627_6.m";
		if (args.length > 0) {
			mfName = args[0];
		}		
		
		String sqFile = "D:\\workspace\\DeepDriver\\bin\\data\\"+mfName;
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

		qlstm.testModel(qsi);
		
	}



}
