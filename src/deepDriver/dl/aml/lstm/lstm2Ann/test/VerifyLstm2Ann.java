package deepDriver.dl.aml.lstm.lstm2Ann.test;

import java.io.File;

import deepDriver.dl.aml.ann.ANN;
import deepDriver.dl.aml.distribution.Fs;
import deepDriver.dl.aml.lrate.StepReductionLR;
import deepDriver.dl.aml.lstm.LSTM;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMWwUpdater;
import deepDriver.dl.aml.lstm.NeuroNetworkArchitecture;
import deepDriver.dl.aml.lstm.apps.wordSegmentation.WordSegSet;
import deepDriver.dl.aml.lstm.lstm2Ann.Lstm2AnnTeacher;

public class VerifyLstm2Ann {
	
	public static void main(String[] args) throws Exception {
		WordSegSet wss = new WordSegSet();
		wss.setMaxLength(1000);
		wss.setRequireBlank(true);
		wss.setRequireEndFlagCheck(false);
		wss.setVoLoadOnly(true); 
		wss.loadWordSegSet("D:\\6.workspace\\p.NLP\\train.conll");
		
		wss.setVoLoadOnly(false); 
		wss.loadWordSegSet("D:\\6.workspace\\p.NLP\\dev.conll");

		int t = 5;
		LSTMConfigurator qcfg = new LSTMConfigurator();
//		qcfg.setBiDirection(true);
		
		qcfg.setBinaryLearning(true);
		qcfg.setLoopNum(30);
		qcfg.setAccuracy(0.01);
		qcfg.setMaxTimePeriod(40);
		qcfg.setLearningRate(0.01);//?
		qcfg.setM(0.5);//?
		qcfg.setMBSize(1);
		qcfg.setDropOut(0);
		qcfg.setEnableUseCellAa(true);
		qcfg.setBatchSize4DeltaWw(true);
		qcfg.setUseRmsProp(false);
		qcfg.setUseThinData(true);
		qcfg.setUseBias(true);
		qcfg.setThreadsNum(4);//?
		StepReductionLR slr = new StepReductionLR();
		slr.setStepsCnt(800000);
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
		WordSegWindowStream qsi = new WordSegWindowStream(wss);
		
		NeuroNetworkArchitecture nna = new NeuroNetworkArchitecture();

		nna.setNnArch(new int [] {128, 128});
		
 		nna.setCostFunction(LSTMConfigurator.SOFT_MAX);
 		qcfg.setRequireLastRNNLayer(false);
		qcfg.buildArchitecture(qsi, nna);
		qcfg.setName("seqLstm");
		
		String mfName = "Tws_1470879785314_7.m";
		if (args.length > 0) {
			mfName = args[0];
		}		
		
//		String sqFile = "D:\\workspace\\DeepDriver\\bin\\data\\"+mfName;
		String sf = System.getProperty("user.dir");	
		File dir = new File(sf, "data");
		File sFile = new File(dir, mfName);
		String sqFile = sFile.getAbsolutePath(); 
		
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
		
		String amfName = "Tws_1470879785314_7.m";
		if (args.length > 1) {
			amfName = args[1];
		}		
		
		File asFile = new File(dir, amfName);
		
		String asqFile = asFile.getAbsolutePath();
		
		int kLength = 4;
		ANN ann = (ANN) Fs.readObjFromFile(asqFile);
		
		Lstm2AnnTeacher teacher = new Lstm2AnnTeacher(qcfg, ann);
		teacher.testModel(qsi, false);

	}

}
