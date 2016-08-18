package deepDriver.dl.aml.lstm.lstm2Ann.test;

import deepDriver.dl.aml.ann.ANN;
import deepDriver.dl.aml.costFunction.SoftMax4ANN;
import deepDriver.dl.aml.lrate.StepReductionLR;
import deepDriver.dl.aml.lstm.LSTM;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.NeuroNetworkArchitecture;
import deepDriver.dl.aml.lstm.apps.wordSegmentation.WordSegSet;
import deepDriver.dl.aml.lstm.lstm2Ann.Lstm2AnnTeacher;

public class TestLstm2Ann {
	
	public static void main(String[] args) throws Exception {
		WordSegSet wss = new WordSegSet();
		wss.setMaxLength(1000);
		wss.setRequireBlank(true);
		wss.setRequireEndFlagCheck(false);
		wss.loadWordSegSet("D:\\6.workspace\\p.NLP\\train.conll");
		
		wss.setVoLoadOnly(true); 
		wss.loadWordSegSet("D:\\6.workspace\\p.NLP\\dev.conll");

		int t = 5;
		LSTMConfigurator qcfg = new LSTMConfigurator();
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
		
		int kLength = 4;
		ANN ann = new ANN();
		ann.setName("seqAnn");
		ann.setCf(new SoftMax4ANN());
		ann.buildUp(new int[]{128, kLength});
		
		Lstm2AnnTeacher teacher = new Lstm2AnnTeacher(qcfg, ann);
		teacher.trainModel(qsi, false);

	}

}
