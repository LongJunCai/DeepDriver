package deepDriver.dl.aml.lstm.enDecoder.test;


import deepDriver.dl.aml.distribution.Fs;
import deepDriver.dl.aml.lrate.StepReductionLR;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.NeuroNetworkArchitecture;
import deepDriver.dl.aml.lstm.Seq2SeqLSTM;
import deepDriver.dl.aml.lstm.distribution.Seq2SeqLSTMBoostrapper;
import deepDriver.dl.aml.lstm.distribution.SimpleTask;
import deepDriver.dl.aml.string.ANFixedStreamImpV2;
import deepDriver.dl.aml.string.Dictionary;
import deepDriver.dl.aml.string.NFixedStreamImpV2;
import deepDriver.dl.aml.string.ThinRandomANFixedStreamImpV2;
import deepDriver.dl.aml.string.ThinRandomQNFixedStreamImpV2;

public class Encoder2DecoderSetup implements Seq2SeqLSTMBoostrapper {

	Dictionary dic = new Dictionary();
	@Override
	public void prepareData(boolean isServer) throws Exception {
		
//		dic.loadDicFromFile("D:\\6.workspace\\ANN\\wangfeng_V2.txt");
		dic.setUsageSortedNum(1000);
//		dic.setDebug(true);
		dic.setUseParser(true);
//		dic.loadDicFromFile("D:\\6.workspace\\ANN\\lstm\\ideaInf");
//		if (args.length > 0) {
//			dic.loadDicFromFile(args[0]);
//		} else {
//			dic.loadDicFromFile("D:\\6.workspace\\ANN\\lstm\\talk2015_2016.txt");
//		}		
		dic = (Dictionary) Fs.readObjFromFile("D:\\6.workspace\\ANN\\lstm\\QaModel\\v_1516.m");
		dic.summaryInf();
//		String sqFile = null;//aa    D:\\workspace\\ANN\\data\\q_lstm_1464485024333_0.m
//		if (args.length >= qFile + 1) {
//			sqFile = args[qFile];
//		}

		
	}
	
	final int t = 5;
	LSTMConfigurator qcfg = new LSTMConfigurator();
	LSTMConfigurator acfg = new LSTMConfigurator();
	NFixedStreamImpV2 qsi;
	ANFixedStreamImpV2 asi;
	NeuroNetworkArchitecture nna;
	
	@Override
	public void bootstrap(SimpleTask task, boolean need4Test) throws Exception {		
		prepareData(true);
		qcfg.setBinaryLearning(true);
		qcfg.setLoopNum(30);
		qcfg.setAccuracy(35);
		qcfg.setMaxTimePeriod(40);
		qcfg.setLearningRate(0.1);//?
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
		
		StepReductionLR srm = new StepReductionLR();
		srm.setStepsCnt(30000);
		srm.setReductionRate(0.5);
		srm.setMinLr(0.01);
		qcfg.setSrm(srm);
//		qcfg.setDebug(true);
//		qcfg.setForceComplete(true);
		
	
		
		acfg.setBinaryLearning(true);
		acfg.setLoopNum(10);
		acfg.setAccuracy(40);
		acfg.setMaxTimePeriod(41);
		acfg.setLearningRate(0.1);
		acfg.setM(0);
		acfg.setMBSize(1);
		acfg.setDropOut(0);
		acfg.setEnableUseCellAa(true);
		acfg.setUseThinData(true);
		acfg.setUseBias(true);
		
		acfg.setThreadsNum(4);
		StepReductionLR aslr = new StepReductionLR();
		aslr.setStepsCnt(50000);
		aslr.setReductionRate(0.5);
		aslr.setMinLr(0.001);
		acfg.setLr(aslr);
		
//		NFixedStreamImpV2 qsi = new RandomQNFixedStreamImpV2(dic, t);
		long l = System.currentTimeMillis();
		qsi = new ThinRandomQNFixedStreamImpV2(dic, t, l);
		asi = new ThinRandomANFixedStreamImpV2(dic, t, l);
		
//		LSTMDataSet lds = dic.encodeSamples(20);
		nna = new NeuroNetworkArchitecture();
//		nna.setUseProjectionLayer(true);
//		nna.setNnArch(new int [] {100, 128, 128});
		nna.setNnArch(new int [] {128, 128});
		nna.setCostFunction(LSTMConfigurator.SOFT_MAX);
		
		qcfg.setRequireLastRNNLayer(false);
		qcfg.buildArchitecture(qsi, nna);
		acfg.buildArchitecture(asi, nna);
		

		qcfg.setName("qcfg");
		acfg.setName("acfg");
	}

	public Dictionary getDic() {
		return dic;
	}

	public void setDic(Dictionary dic) {
		this.dic = dic;
	}

	public LSTMConfigurator getQcfg() {
		return qcfg;
	}

	public void setQcfg(LSTMConfigurator qcfg) {
		this.qcfg = qcfg;
	}

	public LSTMConfigurator getAcfg() {
		return acfg;
	}

	public void setAcfg(LSTMConfigurator acfg) {
		this.acfg = acfg;
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

	public NeuroNetworkArchitecture getNna() {
		return nna;
	}

	public void setNna(NeuroNetworkArchitecture nna) {
		this.nna = nna;
	}

	@Override
	public Seq2SeqLSTM getSeq2SeqLSTM() {
		return null;
	}

	@Override
	public void setSeq2SeqLSTM(Seq2SeqLSTM seq2SeqLSTM) {
		
	}


}
