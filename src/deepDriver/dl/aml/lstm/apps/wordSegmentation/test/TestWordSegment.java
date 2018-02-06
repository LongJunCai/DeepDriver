package deepDriver.dl.aml.lstm.apps.wordSegmentation.test;

import deepDriver.dl.aml.lrate.StepReductionLR;
import deepDriver.dl.aml.lstm.LSTM;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.NeuroNetworkArchitecture;
import deepDriver.dl.aml.lstm.apps.wordSegmentation.WordSegSet;
import deepDriver.dl.aml.lstm.apps.wordSegmentation.WordSegmentationStream;

public class TestWordSegment {

	
	static int qFile = 1;
	static int testA = 2;
	
	public static void main(String[] args) throws Exception {
		
		/*Run under**/
//		DistributionEnvCfg.getCfg().set(P2PServer.KEY_SRV_PORT, 8034);
////		DistributionEnvCfg.getCfg().set(P2PServer.KEY_SRV_HOST, "127.0.0.1");
//		ResourceMaster rm = ResourceMaster.getInstance();
//		if (args != null && args.length > 2) {
//			rm.setup(Integer.parseInt(args[0]));
//		} else {
//			rm.setup(4);
//		}
		
		WordSegSet wss = new WordSegSet();
		wss.loadWordSegSet("D:\\6.workspace\\p.NLP\\train.conll");
		
		wss.setVoLoadOnly(true); 
		wss.loadWordSegSet("D:\\6.workspace\\p.NLP\\dev.conll");

		int t = 5;
		LSTMConfigurator qcfg = new LSTMConfigurator();
		//
		qcfg.setBiDirection(true);
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
		WordSegmentationStream qsi = new WordSegmentationStream(wss);
		
		NeuroNetworkArchitecture nna = new NeuroNetworkArchitecture();

		nna.setNnArch(new int [] {50, 128, 128});
		nna.setUseProjectionLayer(true);
//		nna.setNnArch(new int [] {128, 128});

//		nna.setNnArch(new int [] {256, 256});
		nna.setCostFunction(LSTMConfigurator.SOFT_MAX);
		qcfg.buildArchitecture(qsi, nna);
		
//		if (sqFile != null) {
//			System.out.println("Upgrade qCfg from file: "+sqFile);
//			LSTMWwUpdater wWUpdater = new LSTMWwUpdater(false, true);
//			LSTMWwUpdater deltaWwUpdater = new LSTMWwUpdater(false, false);
//			LSTMWwUpdater checker = new LSTMWwUpdater(true, true);
//			LSTMConfigurator cfg = (LSTMConfigurator) Fs.readObjFromFile(sqFile);
//			checker.updatewWs(cfg, qcfg);
//			wWUpdater.updatewWs(cfg, qcfg); 
//			deltaWwUpdater.updatewWs(cfg, qcfg); 
//			System.out.println("Complete the merging..");
//			checker.updatewWs(cfg, qcfg);
//		}

		qlstm.trainModel(qsi);
		
	}



}
