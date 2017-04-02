package deepDriver.dl.aml.dnc.test.babi;

import java.util.Map;

import deepDriver.dl.aml.ann.ANN;
import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.ann.imp.LogicsticsActivationFunction;
import deepDriver.dl.aml.costFunction.SoftMax4ANN;
import deepDriver.dl.aml.dnc.DNC;
import deepDriver.dl.aml.dnc.DNCConfigurator;
import deepDriver.dl.aml.lrate.StepReductionLR;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.NeuroNetworkArchitecture;
import deepDriver.dl.aml.lstm.imp.TanhAf;
import deepDriver.dl.aml.math.MathUtil;
import deepDriver.dl.aml.string.Dictionary;

public class TestBabi {
	
	public static void main(String[] args) throws Exception {
		Dictionary dic = new Dictionary(); 		
		dic.setUseParser(true);
		String dfile = "D:\\yk.workspace\\p.AI\\babi\\" +
				"tasks_1-20_v1-2\\en-10k\\qa1_single-supporting-fact_train.txt";
		if (args.length > 0) {
			dfile = args[0];			
		}
		
		int threadNum = 4;
		int ldecayLoop = 40000;
		if (args.length > 1) {
			threadNum = Integer.parseInt(args[1]);
		}
		
		if (args.length > 2) {
			ldecayLoop = Integer.parseInt(args[2]);
		}
		
		dic.loadDicFromFile(dfile);
		System.out.println("Loading file from "+dfile);
		
		dic.summary();
		dic.summaryInf();
		Map<String, Integer> mp= dic.getStrMap();
		System.out.println();
		
//		BabiStream babiStream = new BabiStream(dic, 10);
		FullPBabiStream babiStream = new FullPBabiStream(dic, 10);
//		while (babiStream.hasNext()) {
//			babiStream.next(); 
//		}
		LSTMConfigurator cfg = new LSTMConfigurator();
//		qcfg.setBiDirection(true);
		
		int maxTime = 200;
		
//		cfg.setBinaryLearning(true);
		cfg.setLoopNum(30);
		cfg.setAccuracy(0.01);
		cfg.setMaxTimePeriod(maxTime);
		cfg.setLearningRate(0.01);//?
		cfg.setM(0.5);//?
		cfg.setMBSize(1);
		cfg.setDropOut(0);
		cfg.setEnableUseCellAa(true);
		cfg.setBatchSize4DeltaWw(true);
		cfg.setUseRmsProp(false);
//		cfg.setUseThinData(true);
		cfg.setUseBias(true);
		cfg.setThreadsNum(threadNum);//?
		StepReductionLR slr = new StepReductionLR();
		slr.setStepsCnt(800000);
		slr.setReductionRate(0.5); 
		slr.setMinLr(0.001);
		cfg.setLr(slr); 
		
		StepReductionLR srm = new StepReductionLR();
		srm.setStepsCnt(30000);
		srm.setReductionRate(0.5);
		srm.setMinLr(0.01);
		cfg.setSrm(srm);
		cfg.setName("Tws");
		
		NeuroNetworkArchitecture nna = new NeuroNetworkArchitecture();

		nna.setNnArch(new int [] {128, 128});
		int rhNum = 2;
		int memoryNum = 90;
		int memoryLength = 32; 
		
 		nna.setCostFunction(LSTMConfigurator.SOFT_MAX);
 		cfg.setRequireLastRNNLayer(false);
// 		cfg.buildArchitecture(babiStream, nna);
 		cfg.buildArchitecture(babiStream.getTargetFeatureNum(), 
 				babiStream.getSampleFeatureNum() + rhNum * memoryLength, 
 				nna);
 		cfg.setName("dncController");
		
// 		int yLen = rhNum * memoryLength + nna.getNnArch()[nna.getNnArch().length - 1] * 2;
 		int yLen = rhNum * memoryLength + nna.getNnArch()[nna.getNnArch().length - 1];// * nna.getNnArch().length
// 				+ babiStream.getSampleFeatureNum() + rhNum * memoryLength
 				;
// 		int yLen = 512;
		int kLength = babiStream.getTargetFeatureNum();
		ANN ann = new ANN() 
//		{
//			public IActivationFunction createActivation() {
//				return new TanhAf();
//			}
//		}
				;
		ann.getaNNCfg().setThreadsNum(cfg.getThreadsNum());
		ann.setName("dncOutput");
		ann.setCf(new SoftMax4ANN());
		ann.buildUp(new int[]{yLen, kLength});	
		MathUtil.setThreadCnt(1);
		 
		DNCConfigurator dcfg = new DNCConfigurator(0.1, 0.1, maxTime, ann, cfg, yLen, rhNum, memoryNum, memoryLength);
//		dcfg.setL(0.01);
//		dcfg.setM(0.1); 
		dcfg.setMl(0.0001);
		dcfg.setLdecayLoop(ldecayLoop);
		
		DNC dnc = new DNC(dcfg);
		dnc.setFlexL(true);
		dnc.train(babiStream);
		
	}

}
