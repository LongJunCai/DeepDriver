package deepDriver.dl.aml.lstm.apps.pos;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.zip.GZIPInputStream;

import deepDriver.dl.aml.distribution.Fs;
import deepDriver.dl.aml.lrate.StepReductionLR;
import deepDriver.dl.aml.lstm.LSTM;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMWwUpdater;
import deepDriver.dl.aml.lstm.NeuroNetworkArchitecture;
import deepDriver.dl.aml.lstm.apps.util.Embedding;
import deepDriver.dl.aml.lstm.apps.util.GZIPUtil;
import deepDriver.dl.aml.lstm.apps.util.StringUtils;
import deepDriver.dl.aml.lstm.apps.util.TaggedWord;

/**
 * LSTM/BI-LSTM based Part-Of-Speech tagger
 * Training 
 * Java API Usage:
 * Set up the .properties file with parameters(e.g. pos_tagger_default.properties)
 * <p> trainFile, devFile, testFile: dataset path
 * <p> conll: true if input file is in .conll format, false if in plain text format [word/tag,..]
 * <p> biDirection: true for BI-LSTM model, false for LSTM model
 * <p> binaryLearning: true for one-hot input, false for word_embedding input
 * 
 * Command Line Interface Usage:
 * java -cp posTagger.jar deepDriver.dl.aml.lstm.apps.pos.PosTagger \
    -trainFile ${PWD}/train.txt -devFile ${PWD}/dev.txt -testFile ${PWD}/test.txt \
    -dictFile ${PWD}/dict.txt.gz -conll false -biDirection false \ 
    -model 'your_model_name'
 * 
 * Prediction
 * Tag Raw String separated by blanks
 * New Instance of PosTagger: {@link #PosTagger(Properties)}
 * Property file with Model Path(.m), dictFile (.txt.gz) and tSize and fSize are necessary
 * Use below method: {@link #predict(String)}
 * 
 * @since 2016-09-09
 * @author xichen ding
 **/

public class PosTagger {
	
	private static String DEFAULT_ENCODE = "utf-8";
	
	int fSize; // Input Feature size of samples
	int tSize; // Output Target size
	String T_UNKNOWN = "*";
	
	Map<String, Integer> strMap = new HashMap<String, Integer>();
	Map<String, Integer> tagMap = new HashMap<String, Integer>();
	
	LSTM nn;
	
	public PosTagger(){
		try {
			fSize = 0;
			tSize =0;
			nn = null;
		} catch (Exception e) {
			System.out.println(e);
		}
	}
	
	public PosTagger(Properties prop) {
		try {
			String dictFile = prop.getProperty("dictFile");
			readDictFile(dictFile);
			nn = createModel(prop, tSize, fSize);
		} catch (Exception e) {
			System.out.println(e);
		}
	}
	
	/**
	 * Main Access to Training
	 * @param Properties file for input parameters
	 * @return List<TaggedWord> 
	 * */
	
	public static void main(String[] args) throws Exception {
		Properties prop = new Properties();
		// Default Command Line Interface for args
		prop = StringUtils.parseArgs(args);
		if (prop.containsKey("properties")) {
			String pFile = prop.getProperty("properties");
			InputStream in = new BufferedInputStream(new FileInputStream(pFile));
			prop.load(in);
		}
		
		// Java API Usage
		boolean local = true;
		if (local) {
			String pFile = "properties/pos_tagger_default.properties";
			InputStream in = new BufferedInputStream(new FileInputStream(pFile));
			prop.load(in);
		}
		
		// Train new Models
		PosTagger tagger = new PosTagger();
		tagger.trainModel(prop);
	}
	
	private void readDictFile(String dictFile) {
		// Update strMap (<K,V> word, id) and tagMap
		int wordNum = 0;
		int tagNum = 0;
		
		BufferedReader reader = null;
		try {
			if (dictFile.contains(".gz")) {//
				reader = new BufferedReader(new InputStreamReader(
						new GZIPInputStream(new FileInputStream(dictFile)),
						DEFAULT_ENCODE));
			} else {
				reader = new BufferedReader(new InputStreamReader(
						new FileInputStream(new File(dictFile)), DEFAULT_ENCODE));
			}
			String line = null;
			int idxCol = 0;
			while ((line = reader.readLine()) != null) {
				String[] items = line.split("\t");
				String k = items[0];
				Integer v = Integer.valueOf(items[1]);
				if (idxCol == 0)
					wordNum = Integer.valueOf(items[1]);
				else if (idxCol == 1)
					tagNum = Integer.valueOf(items[1]);
				else
					if (idxCol < (wordNum + 2))  // Word Map
						strMap.put(k, v);
					else  // Tag Map
						tagMap.put(k, v);
				idxCol++;
			}
			// Update feature size and target size
			fSize = wordNum;
			tSize = tagNum;
		} catch (Exception e) {
			System.out.println("Errors loading dictionary");
		}
	}
	
	private Map<Integer, String> swapMap(Map<String, Integer> m1) {
		Map<Integer, String> m2 = new HashMap<Integer, String>();
		for (String key:m1.keySet()) {
			m2.put(m1.get(key), key);
		}
		return m2;
	}
	
	/**
	 * @param Properties prop, int tSize, int fSize
	 * Properties contains the basic LSTM config 
	 * tSize: target output size
	 * fSize: input feature size
	 * 
	 * @return LSTM model
	 * */
	
	public LSTM createModel(Properties prop, int tSize, int fSize) throws Exception {
		// String sqFile check continue training or new start
		String sqFile = prop.getProperty("sqFile");
		// Model Configuration Setup
		final LSTMConfigurator qcfg = new LSTMConfigurator();
		// LSTM or BI-LSTM
		boolean bi = (prop.getProperty("biDirection")==null)?false:
			Boolean.valueOf(prop.getProperty("biDirection"));
		qcfg.setBiDirection(bi);
		// Binary Learning:true-One Hot Embedding, false-Word Embedding
		boolean binaryLearning = (prop.getProperty("binaryLearning")==null)?true:
			Boolean.valueOf(prop.getProperty("binaryLearning"));
		qcfg.setBinaryLearning(binaryLearning);
		qcfg.setLoopNum(30);
		qcfg.setAccuracy(1);
		qcfg.setMaxTimePeriod(30); // BPTT Time Period
		qcfg.setLearningRate(0.01);//?
		qcfg.setM(0.1);//?
		qcfg.setMBSize(1);
		qcfg.setDropOut(0.1); // Avoid Overfitting
		qcfg.setEnableUseCellAa(true);
		qcfg.setBatchSize4DeltaWw(true);
		qcfg.setUseRmsProp(false);
		qcfg.setUseThinData(true);
		boolean useBias = (prop.getProperty("useBias")==null)?true:
			Boolean.valueOf(prop.getProperty("useBias"));
		qcfg.setUseBias(useBias);
		qcfg.setThreadsNum(1);//? 1
		StepReductionLR slr = new StepReductionLR();
		slr.setStepsCnt(80000);
		slr.setReductionRate(0.5); 
		slr.setMinLr(0.001);
		qcfg.setLr(slr); 
		
		StepReductionLR srm = new StepReductionLR();
		srm.setStepsCnt(30000);
		srm.setReductionRate(0.5);
		srm.setMinLr(0.01);
		qcfg.setSrm(srm);
		// Model Name
		String name = (prop.getProperty("name")==null)?"model":prop.getProperty("name");
		qcfg.setName(name);
		
		LSTM qlstm = new LSTM(qcfg);
		NeuroNetworkArchitecture nna = new NeuroNetworkArchitecture();
		
		// Default Two hidden layers with 128 nodes each
		nna.setNnArch(new int [] {128, 128});
		nna.setCostFunction(LSTMConfigurator.SOFT_MAX);		
		//Build Layer Architecture, [fSize-128-128-tSize]
		qcfg.buildArchitecture(tSize, fSize, nna);
		
		// Check if continue training from existing models
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
		
		//Log
		System.out.println("#### LSTM Model Config");
		System.out.println("BiDirection: " + bi);
		System.out.println("sqFile: " + sqFile);
		System.out.println("model name: " + name);
		return qlstm;
	}
	
	private PosDataLoader loadCorpus(String trainFile, String devFile, boolean conll) {
		PosDataLoader corpus = new PosDataLoader();
		if (!conll) { // Combining Vocabulary/Tags of train and dev
			corpus.loadDataLine(trainFile);
			corpus.loadDataLine(devFile);
		} else {
			corpus.loadDataConll(trainFile, true);
			corpus.loadDataConll(devFile, true);
		}
		return corpus;
	}
	
	private void updateDataLoader(PosDataLoader corpus, PosDataLoader loader) {
		loader.setStrMap(corpus.getStrMap());
		loader.setIntMap(corpus.getIntMap());
		loader.setTagMap(corpus.getTagMap());
		loader.setWordNum(corpus.getWordNum());
	}
	
	public void saveDictionary(String fileName, Map<String, Integer> strMap,
			Map<String, Integer> tagMap) {
		// strMap, tagMap
		int wordNum = strMap.size();
		int tagNum = tagMap.size();

		try {
			FileOutputStream fileout = new FileOutputStream(new File(fileName));
			OutputStreamWriter writer = new OutputStreamWriter(fileout,
					DEFAULT_ENCODE);
			writer.write("wordNum" + "\t" + wordNum + "\n");
			writer.write("tagNum" + "\t" + tagNum + "\n");
			for (String k1 : strMap.keySet())
				writer.write(k1 + "\t" + strMap.get(k1) + "\n");
			for (String k2 : tagMap.keySet())
				writer.write(k2 + "\t" + tagMap.get(k2) + "\n");
			writer.close();
		} catch (Exception e) {

		}
		// Compress File
		GZIPUtil.compressFile(fileName);
	}
	
	public PosStream loadDataset(Properties prop, String choice){
		return loadDataset(prop, choice, true);
	}
	
	public PosStream loadDataset(Properties prop, String choice, boolean verbose){
		String trainFile = null;
		String devFile = null;
		String testFile = null;
		String embedFile = null; 
		String model = null;
		Integer embeddingSize;
		boolean conll = false;
		if (prop.containsKey("trainFile")){trainFile = prop.getProperty("trainFile");}
		if (prop.containsKey("devFile")){devFile = prop.getProperty("devFile");}
		if (prop.containsKey("testFile")){testFile = prop.getProperty("testFile");}
		if (prop.containsKey("embedFile")){embedFile = prop.getProperty("embedFile");}
		if (prop.containsKey("model")){model = prop.getProperty("model");}
		if (prop.containsKey("conll")){conll = Boolean.valueOf(prop.getProperty("conll"));}
		// Default embedding input dimension 50
		embeddingSize = (prop.getProperty("embeddingSize")==null)?50:
			Integer.valueOf(prop.getProperty("embeddingSize"));
		
		// Log
		if (verbose) {
			System.out.println("#### Reading Model Properties");
			System.out.println("trainFile:" + trainFile);
			System.out.println("devFile:" + devFile);
			System.out.println("testFile:" + testFile);
			System.out.println("embedFile:" + embedFile);
			System.out.println("embeddingSize:" + embeddingSize);
			System.out.println("model:" + model);
			System.out.println("conll:" + conll);			
		}
		
		Embedding embedding = new Embedding();
		PosDataLoader train = new PosDataLoader();
		PosDataLoader dev = new PosDataLoader();
		PosDataLoader test = new PosDataLoader();
		PosDataLoader corpus = loadCorpus(trainFile, devFile, conll);
		
		// Save Dictionary Files, aaa.txt.gz
		saveDictionary(prop.getProperty("dictFile").replace(".gz", ""), corpus.getStrMap(), corpus.getTagMap());
		
		if (!conll) {//input format is plain text
			train.loadDataLine(trainFile);
			dev.loadDataLine(devFile);
			test.loadDataLine(testFile);
		} else {//input format is Conll
			train.loadDataConll(trainFile, false);
			dev.loadDataConll(devFile, false);
			test.loadDataConll(testFile, false);
		}
		updateDataLoader(corpus, train); // Combine words/tags train/dev combination
		updateDataLoader(corpus, dev);
		updateDataLoader(corpus, test);
		
		if (verbose) {
			System.out.println("#### "+"Training Samples:" + train.getSentences().size());
			System.out.println("#### "+"Dev Samples:" + dev.getSentences().size());
			System.out.println("#### "+"Test Samples:" + test.getSentences().size());			
		}

		PosStream ps = null;
		if (choice.equals("train")) {
			if (embedFile!=null) {
				System.out.println("Embedding Files Used");
				embedding.loadData(embedFile);
				ps = new PosStream(train, embedding);
			} else {
				System.out.println("Word One Hot Encoding Used");
				ps = new PosStream(train);
			}			
		} else if (choice.equals("dev")) {
			if (embedFile!=null) {
				System.out.println("Embedding Files Used");
				embedding.loadData(embedFile);
				ps = new PosStream(dev, embedding);
			} else {
				System.out.println("Word One Hot Encoding Used");
				ps = new PosStream(dev);
			}
		} else if (choice.equals("test")) {
			if (embedFile!=null) {
				System.out.println("Embedding Files Used");
				embedding.loadData(embedFile);
				ps = new PosStream(test, embedding);
			} else {
				System.out.println("Word One Hot Encoding Used");
				ps = new PosStream(test);
			}
		}
		return ps;
	}
	
	public void trainModel(Properties prop) throws Exception {
		// Load train POS stream
		PosStream ps = loadDataset(prop, "train");
		// Model Configuration
		fSize = ps.getSampleFeatureNum();
		tSize = ps.getTargetFeatureNum();
		System.out.println("#### "+"Input Feature Number:" + fSize);
		System.out.println("#### "+"Output Target Number:" + tSize);
		nn = createModel(prop, tSize, fSize); // Creating New Model
		nn.trainModel(ps);
		
	}
	
	/**
	 * @param String raw text separated by blanks
	 * @return List<TaggedWord> as tagging results 
	 * */
	
	public List<TaggedWord> predict(String doc){
		String[] words = doc.split(" ");
		int size = words.length;
		double[][] sampleTT = new double[size][];
		for (int i = 0; i < size; i++) {
			String word = words[i];
			// one hot input
			sampleTT[i] = new double[1];
			Integer idx = strMap.get(word);
			if (idx == null) { // Uknown Character
				idx = strMap.get(T_UNKNOWN);
			}
			sampleTT[i][0] = Double.valueOf(idx);
		}
		
		int[] posIdx = null;
		// To Do: Add predict method to LSTM, 
		// return the predict index of target class
		// int[] posIdx = nn.predict(sampleTT);
		
		Map<Integer, String> idxMap = swapMap(tagMap);
		List<TaggedWord> sentence = new ArrayList<TaggedWord>();
		for (int i = 0; i < size; i++) {
			String tag = idxMap.get(posIdx[i]);
			sentence.add(new TaggedWord(words[i], tag));
		}
		return sentence;
	}

}
