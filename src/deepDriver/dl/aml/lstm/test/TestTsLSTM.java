package deepDriver.dl.aml.lstm.test;


import deepDriver.dl.aml.ann.Normalizer;
import deepDriver.dl.aml.lstm.LSTM;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMDataSet;
import deepDriver.dl.aml.lstm.NeuroNetworkArchitecture;

public class TestTsLSTM {
	
	public static void main(String[] args) {
		double [] ts = new double[] {
				40,43.545,47.214,50.518,52.959,
				54.102,53.639,51.449,47.621,42.47,36.506,30.392,24.864,20.646,18.356,18.424,21.022,26.027,33.015,41.295,49.977,58.067,64.579,68.656,69.681,67.358,61.772,53.394,43.044,31.816,20.959,11.741,5.299,2.5,3.833,9.327,18.537,30.57,44.172,57.857,70.071,79.368,84.577,84.954,80.276,70.889,57.687,42.03,25.605,10.242,
				//hua qian gu
//				 32408808.00 ,49065305.00 ,35840339.00 ,29463679.00 ,24995956.00 ,17526739.00 ,16790912.00 ,68472777.00 ,81590319.00 ,49535212.00 ,41661755.00 ,34093391.00 ,29455636.00 ,25756008.00 ,88940334.00 ,102794078.00 ,58135311.00 ,46138043.00 ,36679363.00 ,28448170.00 ,29931288.00 ,95311010.00 ,113181991.00 ,60660788.00 ,49781625.00 ,
				//lai zi xingxing de ni
//				2129979,2938754,1992125,1602018,1173308,803996,1059289,3981570,4233244,3517679,2731921,1983610,1663027,1187353,3426353,7625907,5562293,4381950,3423598,2496718,3057295,9108776,11903951,8516172,7265244,5853051,5441012
//				2129979,2938754,1992125,1602018,1173308,803996,1059289,3981570,4233244,3517679,2731921,1983610,1663027,1187353,3426353,7625907,5562293,4381950,3423598,2496718,3057295,9108776,11903951,8516172,7265244,5853051,5441012,5502924,12245989,14416734,10138954,8623638,7265652,6848352,6728144,16057976,19962624,13144691,11068960
//				2129979,2938754,1992125,1602018,1173308,803996,1059289,3981570,4233244,3517679,2731921,1983610,1663027,1187353,3426353,7625907,5562293,4381950,3423598,2496718,3057295,9108776,11903951,8516172,7265244,5853051,5441012,5502924,12245989,14416734,10138954,8623638,7265652,6848352,6728144,16057976,19962624,13144691,11068960,9766096,9197498,9079688,13905275,13853288,10595127,9709504,9466805,9232267,9694626,20795653,26016776,19146779,18246221,15809689,15646925,19201830,35674144,39708292,30171168,27338290,22373139,22432686,24324198
				
				//lang ya bang
//				865734.00 ,3261285.00 ,
//				17996115.00 ,21637383.00 ,26800431.00 ,30341182.00 ,35420800.00 ,39179240.00 ,44002801.00 ,47542311.00 ,48372504.00 ,52439973.00 ,54488587.00 ,56850284.00 ,57007083.00 ,62516621.00 ,66068447.00 ,68171854.00 ,71615398.00 ,79403162.00 ,80135254.00 ,82628070.00 ,85343471.00 ,92548889.00 ,87881117.00 ,
//				25489904,29722274, 35017786,32558119,26925755,26957702,28302980,28335308,30468169,35337236,31572531,28847576,30386635,34035088,37208254,36012169,36866488,37597189,37664531,38253655,38574566,32378707,31814827,33768886,36049622,29023295,
		};
		Normalizer normalizer = new Normalizer();
//		normalizer.setMaxPeak(500);
//		normalizer.setMaxPeak(91068960);
		double [] tts = normalizer.transformParameters(ts);
		
		int flength = 2;
		int tlength = flength;
		int klength = 2;	
		int t = 11;
		int sampleNum = tts.length/t;		
		double [][][] samples = new double[sampleNum][][];		
		double [][][] targets = new double[sampleNum][][];
		for (int i = 0; i < samples.length; i++) {
			samples[i] = new double[t][flength]; 
			targets[i] = new double[t][tlength];  
			double [][] sample = samples[i];
			double [][] target = targets[i];
			for (int j = 0; j < sample.length; j++) {
				sample[j] = new double[flength];
				target[j] = new double[tlength];
				for (int j2 = 0; j2 < flength; j2++) {
					sample[j][j2] = tts[i*t + j + j2];
				}
				for (int j2 = 0; j2 < tlength; j2++) { 
					target[j][j2] = tts[i*t + j + j2 + klength];
				}
//				target[j][0] = tts[i*t + j + flength];
			}
		}
		double [][][] samples1 = new double[sampleNum][][];
		for (int i = 0; i < samples1.length; i++) {
			samples1[i] = new double[t][flength];  
			double [][] sample = samples1[i]; 
			for (int j = 0; j < sample.length; j++) {
				sample[j] = new double[flength];
				for (int j2 = 0; j2 < flength; j2++) {
					sample[j][j2] = tts[i*t + j + j2];
				}				
			}
		}
		LSTMConfigurator cfg = new LSTMConfigurator();
		LSTMDataSet ds = new LSTMDataSet();
		ds.setSamples(samples);
		ds.setTargets(targets);
		NeuroNetworkArchitecture nna = new NeuroNetworkArchitecture();
		nna.setNnArch(new int [] {8});
		cfg.setM(0.5);
		cfg.setLearningRate(0.01);
		cfg.setLoopNum(100);
		cfg.setAccuracy(0.1);
		cfg.setEnableUseCellAa(true);
//		lstm.setDropOut(0.1);
		cfg.buildArchitecture(ds, nna);
		
		LSTM lstm = new LSTM(cfg);
		lstm.setCleanData(false);
		lstm.trainModel(ds);
		
//		System.out.println("finished training already");
//		ds.setSamples(samples1);
		double [][][] resuts = lstm.testModel(ds, 100);
		
		for (int i = 0; i < resuts.length; i++) {
			double [][] result = normalizer.transformBackParameters(resuts[i]);
			for (int j = 0; j < result.length; j++) {
//				System.out.println(resuts[i][j][0]+","+result[j][0]);
				System.out.println(result[j][0]);
			}			
		}
	}

}
