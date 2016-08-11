package deepDriver.dl.aml.lstm.test;


import deepDriver.dl.aml.lstm.LSTM;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMDataSet;
import deepDriver.dl.aml.lstm.NeuroNetworkArchitecture;
import deepDriver.dl.aml.string.Dictionary;
import deepDriver.dl.aml.string.NFixedStreamImpV2;

public class TestSongLSTM {
	
	public static void main(String[] args) throws Exception {
		Dictionary dic = new Dictionary();
//		dic.setMaxCnt(150);
		dic.loadDicFromFile("D:\\6.workspace\\ANN\\wangfeng_V2.txt");
//		dic.loadDicFromFile("D:\\6.workspace\\ANN\\wangfeng.txt");

		LSTMConfigurator cfg = new LSTMConfigurator();
		cfg.setName("lstmLM");
		cfg.setBinaryLearning(true);
		cfg.setLoopNum(7);
		cfg.setAccuracy(5);
		cfg.setMaxTimePeriod(40);
		cfg.setLearningRate(0.001);
//		cfg.setM(0.5);
//		cfg.setMBSize(1);	
		cfg.setM(0.5);
		cfg.setMBSize(32);
		cfg.setDropOut(0.0);
		cfg.setEnableUseCellAa(true);
		cfg.setBatchSize4DeltaWw(false);
		int t = 5;
		NFixedStreamImpV2 si = new NFixedStreamImpV2(dic, t);
//		LSTMDataSet lds = dic.encodeSamples(20);
		NeuroNetworkArchitecture nna = new NeuroNetworkArchitecture();
		nna.setNnArch(new int [] {120});
		nna.setCostFunction(LSTMConfigurator.SOFT_MAX);
		System.out.println("Start training...");
		cfg.buildArchitecture(si, nna);
		LSTM lstm = new LSTM(cfg);
//		lstm.trainModel(si); 
		lstm.trainModelWithBatchSize(si);
		int cnt = 1;
		while (lstm.getError() > cfg.getAccuracy()) {
			lstm.trainModelWithBatchSize(si); 
			cnt ++;
			if (cnt == 10) {
				cnt = 0;
				lstm.finish1Cycle();
				System.out.println("Error "+lstm.getError());
			}
		}
		
		
//		String start = "I我真的需要";
		String start = "他";
		LSTMDataSet tds = dic.encodeSample(start, t);
		System.out.println("Start testing..."+start);
		double [][][] ts = lstm.testModel(tds, 200);
		String ss = dic.decoded(ts);
		System.out.println(ss);
	}

}
