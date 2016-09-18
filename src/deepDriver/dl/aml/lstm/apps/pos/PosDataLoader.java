package deepDriver.dl.aml.lstm.apps.pos;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import deepDriver.dl.aml.lstm.apps.util.TaggedWord;

public class PosDataLoader {

	private static String DEFAULT_ENCODE = "utf-8";
	List<List<TaggedWord>> sentences;
	Map<String, Integer> tagMap;
	
	// Vocabulary
	int wordNum = 0;
	int maxLength = 30; // sentencese longer will be truncated
	Map<String, Integer> strMap;
	Map<Integer, String> intMap;
	
	String T_UNKNOWN = "*"; // Unknown Tokens
	String T_BOS = "BOS"; // Beginning of Sentences
	String T_EOS = "EOS"; // End of Sentences
	
	public PosDataLoader() {
		sentences = new ArrayList<List<TaggedWord>>();
		tagMap = new HashMap<String, Integer>();
		strMap = new HashMap<String, Integer>();
		strMap.put(T_UNKNOWN, 0);
		intMap = new HashMap<Integer, String>();
		intMap.put(0, T_UNKNOWN);
		wordNum++; // unknown
	}
	
	/**
	 * Reading Data File in .conll format
	 * */
	
	public void loadDataConll(String file, boolean vocab) {
		List<TaggedWord> sen = new ArrayList<TaggedWord>();
		int cnt = 0; // tag Count
		try {
			BufferedReader bi = new BufferedReader(new InputStreamReader(
					new FileInputStream(new File(file)), DEFAULT_ENCODE));
			String line = null;
			
			while ((line=bi.readLine()) != null) {
				if (line.trim().length() == 0) {
					// End of cur sentences TAB Line
					if (sen.size() > 0) { 
					    sentences.add(sen);
					}
					sen = new ArrayList<TaggedWord>();
					continue;
				}
				
				String[] items = line.split("\t");
				sen.add(new TaggedWord(items[2], items[3]));
				
				// Update Vocabulary
				if (vocab) {
					if (!strMap.containsKey(items[2])) {
						strMap.put(items[2], wordNum);
						intMap.put(wordNum, items[2]);
						wordNum++;
					}				
				}
				
				if (!tagMap.containsKey(items[3])) {// index from 0
					tagMap.put(items[3], cnt);
					cnt++;
				}
				
				// Check if sentence need truncated if longer than max
				if (sen.size() >= maxLength) {
					// System.out.println("Sentences Longer than 40 tokens");
					if (sen.size() > 0) { 
					    sentences.add(sen);
					}
					sen = new ArrayList<TaggedWord>();
					continue;
				}
			}
			bi.close();			
			
			
		} catch (Exception e) {
			System.out.println("Errors Loading Data");
		}
		
	}
	
	/**
	 * Reading Data File in plain text line format
	 * */
	
	public void loadDataLine(String file) {
		List<TaggedWord> sen = new ArrayList<TaggedWord>();
		int cnt = 0; // tag Count
		
		try {
			BufferedReader bi = new BufferedReader(new InputStreamReader(
					new FileInputStream(new File(file)), DEFAULT_ENCODE));
			String line = null;

			while ((line=bi.readLine()) != null) {
				if (line.trim().length() == 0) {
					continue;
				}
				String[] items = line.trim().split("\\s+");//multiple blanks
				//e.g. 19980101-01-003-001/m  北京/ns  举行/v
				for (int i = 1; i < items.length; i++) {
					String[] pair = items[i].split("/");
					sen.add(new TaggedWord(pair[0], pair[1]));
					
					// Update Vocabulary
					if (!strMap.containsKey(pair[0])) {
						strMap.put(pair[0], wordNum);
						intMap.put(wordNum, pair[0]);
						wordNum++;
					}
					if (!tagMap.containsKey(pair[1])) {// index from 0
						tagMap.put(pair[1], cnt);
						cnt++;
					}
					// Check if sentence need truncated if longer than max
					if (sen.size() >= maxLength) {
						// System.out.println("Sentences Longer than 40 tokens");
						sentences.add(sen);
						sen = new ArrayList<TaggedWord>();
						continue;
					}
				}
				// Add Sentences
				if (sen.size() > 0) {
					sentences.add(sen);
					sen = new ArrayList<TaggedWord>();
				}
			}
			bi.close();			
		} catch (Exception e) {
			System.out.println("Errors Loading Data");
		}
	}
	
	/**
	 * Split Train and Dev Dataset
	 * */
	
	private List<PosDataLoader> splitCorpus(PosDataLoader corpus) {
		double r1 = 0.8;
		double r2 = 0.85;
		double r3 = 1.0;
		List<PosDataLoader> loaderlist = splitCorpus(corpus, r1, r2, r3);
		return loaderlist;
	}
	
	private List<PosDataLoader> splitCorpus(PosDataLoader corpus, double r1,
			double r2, double r3) {
		PosDataLoader train = new PosDataLoader();
		PosDataLoader dev = new PosDataLoader();
		PosDataLoader test = new PosDataLoader();
		
		int num = corpus.getSentences().size();
		
		int s1 = (int) ((double) num * r1);
		int s2 = (int) ((double) num * r2);
		int s3 = (int) ((double) num * r3);
		
		train.setSentences(corpus.getSentences().subList(0, s1));
		train.setStrMap(corpus.getStrMap());
		train.setTagMap(corpus.getTagMap());
		train.setWordNum(corpus.getWordNum());
		
		dev.setSentences(corpus.getSentences().subList(s1 + 1, s2));
		dev.setStrMap(corpus.getStrMap());
		dev.setTagMap(corpus.getTagMap());
		dev.setWordNum(corpus.getWordNum());
		
		test.setSentences(corpus.getSentences().subList(s2 + 1, s3));
		test.setStrMap(corpus.getStrMap());
		test.setTagMap(corpus.getTagMap());
		test.setWordNum(corpus.getWordNum());		
		
		List<PosDataLoader> list = new ArrayList<PosDataLoader>();
		list.add(train);
		list.add(dev);
		list.add(test);
		return list;
	}
	
	public void saveDictWords(String fileName) throws IOException {
		File file = new File(fileName);
		OutputStreamWriter writer = new OutputStreamWriter(
				new FileOutputStream(file), DEFAULT_ENCODE);
		for (String key : strMap.keySet()) {
			writer.write(key + "\t" + strMap.get(key) + "\n");
		}
		writer.close();
	}
	
	public void saveDictTags(String fileName) throws IOException {
		File file = new File(fileName);
		OutputStreamWriter writer = new OutputStreamWriter(
				new FileOutputStream(file), DEFAULT_ENCODE);
		for (String key : tagMap.keySet()) {
			writer.write(key + "\t" + tagMap.get(key) + "\n");
		}
		writer.close();
	}
	
	public void saveDoc(String fileName) throws IOException {
		File file = new File(fileName);
		OutputStreamWriter writer = new OutputStreamWriter(
				new FileOutputStream(file), DEFAULT_ENCODE);
		for (int i = 0; i < sentences.size(); i++) {
			List<TaggedWord> sen = sentences.get(i);
			for (int j = 0; j < sen.size(); j++) {
				TaggedWord w = sen.get(j);
				writer.write(w.word() + "/" + w.tag() + "  ");
			}
			writer.write("\n");
		}
		writer.close();
	}
	
	public List<List<TaggedWord>> getSentences() {
		return sentences;
	}
	
	public void setSentences(List<List<TaggedWord>> sentences) {
		this.sentences = sentences;
	}

	public Map<String, Integer> getTagMap() {
		return tagMap;
	}

	public void setTagMap(Map<String, Integer> tagMap) {
		this.tagMap = tagMap;
	}

	public Map<String, Integer> getStrMap() {
		return strMap;
	}

	public void setStrMap(Map<String, Integer> strMap) {
		this.strMap = strMap;
	}

	public Map<Integer, String> getIntMap() {
		return intMap;
	}

	public void setIntMap(Map<Integer, String> intMap) {
		this.intMap = intMap;
	}

	public int getWordNum() {
		return wordNum;
	}

	public void setWordNum(int wordNum) {
		this.wordNum = wordNum;
	}
	
	public static void main(String[] args) throws IOException {
//		String file = "D:\\nlp\\corpus\\treebank\\HIT\\train.conll";
//		PosDataLoader loader = new PosDataLoader();
//		loader.loadDataConll(file, true);
//		loader.saveDictTags("D:\\nlp\\corpus\\treebank\\HIT\\dict_tags.txt");
//		loader.saveDictWords("D:\\nlp\\corpus\\treebank\\HIT\\dict_words.txt");
		
		PosDataLoader corpus = new PosDataLoader();
		String file = "D:\\nlp\\corpus\\POS\\china_daily\\199801_utf8_new.txt";
		corpus.loadDataLine(file);
		System.out.println("Corpus " + corpus.getSentences().size());
		
		List<PosDataLoader> list = corpus.splitCorpus(corpus);
		PosDataLoader train = list.get(0);
		PosDataLoader dev = list.get(1);
		PosDataLoader test = list.get(2);
		train.saveDoc("D:\\nlp\\corpus\\POS\\china_daily\\199801_train.txt");
		System.out.println("Train " + train.getSentences().size());
		dev.saveDoc("D:\\nlp\\corpus\\POS\\china_daily\\199801_dev.txt");
		System.out.println("Dev " + dev.getSentences().size());
		test.saveDoc("D:\\nlp\\corpus\\POS\\china_daily\\199801_test.txt");
		System.out.println("Test " + test.getSentences().size());
		
	}
	
}
