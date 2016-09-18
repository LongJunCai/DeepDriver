package deepDriver.dl.aml.lstm.apps.ner.test;

import java.util.Properties;

import deepDriver.dl.aml.lstm.LSTM;
import deepDriver.dl.aml.lstm.apps.ner.NerStream;
import deepDriver.dl.aml.lstm.apps.ner.NerTagger;
import deepDriver.dl.aml.lstm.apps.util.StringUtils;

public class NerTaggerVerify {
	
	static int qFile = 0;
	static int testA = 2;
	
	public static void main(String[] args) throws Exception {
		NerTagger tagger = new NerTagger();
		Properties prop = StringUtils.parseArgs(args);
		
		// Read from local property files
		// String pf =
		// "C:\\workspace\\DeepDriver\\properties\\tagger_default.properties";
		// InputStream in = new BufferedInputStream(new FileInputStream(pf));
		// Properties prop = new Properties();
		// prop.load(in);
		// prop.setProperty("sqFile",
		// "C:\\workspace\\DeepDriver\\data\\china_daily_1472631418124_0.m");
		
		System.out.println("###### Reading Training Datasets ######");
		NerStream psTrain = tagger.loadDataset(prop, "train", false);
		System.out.println("###### Reading Development Datasets ######");
		NerStream psDev = tagger.loadDataset(prop, "dev", false);
		System.out.println("###### Reading Test Datasets ######");
		NerStream psTest = tagger.loadDataset(prop, "test", false);
		
		int tSize = psTrain.getTargetFeatureNum();
		int fSize = psTrain.getSampleFeatureNum();
		final LSTM qlstm = tagger.createModel(prop, tSize, fSize);
		
		if (prop.getProperty("devFile")!=null) {
			System.out.println("###### Evaluate Development Datasets ######");
			qlstm.testModel(psDev);
		}
		if (prop.getProperty("trainFile")!=null) {
			System.out.println("###### Evaluate Training Datasets ######");
			qlstm.testModel(psTrain);
		}
		if (prop.getProperty("testFile")!=null) {
			System.out.println("###### Evaluate Test Datasets ######");
			qlstm.testModel(psTest);
		}
		
	}
	
}
