package deepDriver.dl.aml.cnn2lstm.test;


import deepDriver.dl.aml.cnn.CNNConfigurator;
import deepDriver.dl.aml.cnn.img.CsvImgLoader;
import deepDriver.dl.aml.cnn.img.ImgDataStream;
import deepDriver.dl.aml.cnn.nets.LeNet;
import deepDriver.dl.aml.cnn2lstm.CNN2LSTMTeacher;
import deepDriver.dl.aml.lrate.StepReductionLR;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.NeuroNetworkArchitecture;
import deepDriver.dl.aml.string.Dictionary;
import deepDriver.dl.aml.string.NFixedStreamImpV2;

public class TestCNN2LSTM {
	static int qFile = 1;
	static int testA = 2;
	
	public static void main(String[] args) throws Exception {
		final Dictionary dic = new Dictionary();
		dic.setUsageSortedNum(1000);
		dic.setUseParser(true);
		if (args.length > 0) {
			dic.loadDicFromFile(args[0]);
		} else {
			dic.loadDicFromFile("D:\\6.workspace\\ANN\\lstm\\talk2015_2016.txt");
		}		
		
		String sqFile = null;//aa    D:\\workspace\\ANN\\data\\q_lstm_1464485024333_0.m
		if (args.length >= qFile + 1) {
			sqFile = args[qFile];
		}

		final int t = 5;
		final LSTMConfigurator qcfg = new LSTMConfigurator();
		qcfg.setBinaryLearning(true);
		qcfg.setLoopNum(30);
		qcfg.setAccuracy(35);
		qcfg.setMaxTimePeriod(40);
		qcfg.setLearningRate(0.3);//?
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
		slr.setStepsCnt(50000);
		slr.setReductionRate(0.5);
		slr.setMinLr(0.001);
		qcfg.setLr(slr);
		
		NFixedStreamImpV2 nfixedStreamImpV2 = new NFixedStreamImpV2(dic, 5, 0);
		
		NeuroNetworkArchitecture nna = new NeuroNetworkArchitecture();
		nna.setNnArch(new int [] {128, 128});
		nna.setCostFunction(LSTMConfigurator.SOFT_MAX);
		qcfg.buildArchitecture(nfixedStreamImpV2.getTargetFeatureNum(), 
				nfixedStreamImpV2.getSampleFeatureNum() + 120, nna);
		
		StepReductionLR srm = new StepReductionLR();
		srm.setStepsCnt(30000);
		srm.setReductionRate(0.5);
		srm.setMinLr(0.01);
		qcfg.setSrm(srm);
		
		CNNConfigurator cnnCfg = new CNNConfigurator(); 
		cnnCfg.setL(0.001);
		cnnCfg.setName("I2W_CNN");
		
		LeNet leNet = new LeNet();
		leNet.constructNet(cnnCfg);	
		
		CNN2LSTMTeacher cnn2LSTMTeacher = new CNN2LSTMTeacher(cnnCfg, qcfg);
		
		CsvImgLoader imgLoader = new CsvImgLoader();
		imgLoader.loadImg("\\\\\\\\\\");//
		int kLength = 10;
		ImgDataStream imgDataStream = new ImgDataStream(imgLoader, kLength);
		cnn2LSTMTeacher.trainModel(imgDataStream, nfixedStreamImpV2, false);
	}

}
