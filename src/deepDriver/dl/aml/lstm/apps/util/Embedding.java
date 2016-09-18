package deepDriver.dl.aml.lstm.apps.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Embedding {
	
	public int dim; // dimension
	public int vSize; // vocabulary size
	
	double[][] matrix; // embedding matrix
	// <K,V> word, index
	Map<String, Integer> wordMap = new HashMap<String, Integer>();
	// <K,V> indx, wor
	Map<Integer, String> idxMap = new HashMap<Integer, String>();
	
	// Load Embedding Files
	public void loadData(String file) {
		int idx = 0;
		List<double[]> data = new ArrayList<double[]>();
		
		try {
			BufferedReader bi = new BufferedReader(new InputStreamReader(
					new FileInputStream(new File(file)), "utf-8"));
			String line = bi.readLine();
			while (line != null) {
				String[] items = line.split("\t");
				
				if (!wordMap.containsKey(items[0])) {
					wordMap.put(items[0], idx);
					idxMap.put(idx, items[0]);
					idx++; // index start from 0
				}
				
				dim = items.length - 1;
				double[] vec = new double[dim];
				for (int i = 0; i < dim; i++) {
					vec[i] = Double.valueOf(items[i+1]);
				}
				data.add(vec);
				line = bi.readLine();
			}
			bi.close();

			vSize = data.size();
			matrix = new double[vSize][];
			
			for (int i = 0; i < vSize; i++) {
				matrix[i] = data.get(i);
			}
		} catch (Exception e) {
			System.out.println("Errors Reading Embedding Files");
		}
	}
	
	public double[] getWordVec(String word){
		int id; 
		if (!wordMap.containsKey(word)) {
			// unknown character
			return new double[dim];
		} else {
			id = wordMap.get(word);
			return matrix[id];
		}		
	}
	
	public int getWordId(String word) {
		int id = 0; 
		if (!wordMap.containsKey(word)) {
			// unknown character
		} else {
			id = wordMap.get(word);
		}
		return id;
	}
	
	public static void main(String[] args){
		String file = "D:\\nlp\\corpus\\Chinese\\chinese_embedding_v50_HIT.txt";
		Embedding embedding = new Embedding();
		embedding.loadData(file);
		System.out.println(embedding.vSize);
	}
	
}
