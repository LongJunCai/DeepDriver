package deepDriver.dl.aml.w2v;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import deepDriver.dl.aml.math.MathUtil;

public class W2V implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	HashMap<String, double []> w2v = new HashMap<String, double []>();
	HashMap<String, Integer> wCnt = new HashMap<String, Integer>();
	
//	HashMap<String, Integer> wCnt = new HashMap<String, Integer>();
	
	static transient Random random = new Random(System.currentTimeMillis());
	
	
	
	List<KeyCntPair> keyCntPairs = new ArrayList<KeyCntPair>();
	List<KeyCntPair> allKeyCntPairs = new ArrayList<KeyCntPair>();
	
	public String hit(double r) {
		double l = r * (double)negRepeatCnt;
		double lc = 0;
		for (int i = 0; i < keyCntPairs.size(); i++) {
			lc = lc + keyCntPairs.get(i).value;
			if (lc >= l) {
				return keyCntPairs.get(i).key;
			}
		}
		return null;
	}
	
	int allRepeatCnt = 0;
	int wordSegCnt = 0;
	
	int lowThreadhold = 5;
	double highThreadhold = 0.001;
	
	int negRepeatCnt = 0;
	
	public void etlData() {
		Iterator<String> iter = wCnt.keySet().iterator();
		while (iter.hasNext()) {
			String key = (String) iter.next();
			KeyCntPair kcp = new KeyCntPair();
			kcp.key = key;
			kcp.value = wCnt.get(key);		
			allKeyCntPairs.add(kcp);
			wordSegCnt ++;
			
			//remove the low/high frequence words.
			if (kcp.value <= lowThreadhold) {
				continue;
			}
			double f = kcp.value/(double)allRepeatCnt;
			double f1 = highThreadhold/f;
			double ran = Math.sqrt(f1) + f1;
			double r = random.nextDouble();
			if (r >= ran) {
				continue;
			}
			keyCntPairs.add(kcp);
			negRepeatCnt = negRepeatCnt + (int)kcp.value;
			
		}
		sort(keyCntPairs);
	}
	
	public void sort(List<KeyCntPair> kcps) {		
		KeyCntPair tkcp = new KeyCntPair();
		for (int i = 0; i < kcps.size(); i++) {
			KeyCntPair kcpi = kcps.get(i);
			for (int j = i + 1; j < kcps.size(); j++) {
				KeyCntPair kcpj = kcps.get(j);
				if (kcpi.value < kcpj.value) {
					switchKcp(kcpi, tkcp);
					switchKcp(kcpj, kcpi);
					switchKcp(tkcp, kcpj);
				}
			}
			if (debug) {
				System.out.println(kcpi.key + "," + kcpi.value);
			}
//			
		}
		
		summary();
	}
	
	public void summary() {
		System.out.println("There are "+wordSegCnt+" unique words, " +
				"and there are "+allRepeatCnt+" words used.");
	}
	
	boolean debug;
	
	public void switchKcp(KeyCntPair s, KeyCntPair t) {
		t.key = s.key;
		t.value = s.value;
	}
	
	public double [] getByWord(String w) {
		return w2v.get(w);
	}
	
	public void put(String w, double [] v) {
		w2v.put(w, v);
	}
	
	public void freshCnt(String w) {
		if (wCnt.get(w) == null) {
			wCnt.put(w, 1);
		} else {
			wCnt.put(w, wCnt.get(w) + 1);
		}		
		allRepeatCnt ++; 
	}
	
	int projectionLength = 100;
	
	
	public int getProjectionLength() {
		return projectionLength;
	}

	public void setProjectionLength(int projectionLength) {
		this.projectionLength = projectionLength;
	}
	
	double min = -1.0;
	double max = 1.0;
	double length = max - min; 
	
	double bound = 0.5;

	public double [] generateV() {
		double [] v = new double[projectionLength];
		for (int i = 0; i < v.length; i++) {
//			v[i] = length * random.nextDouble()
//					+ min;
			v[i] = (random.nextDouble()/max - bound)/(double)projectionLength;
		}
		return v;
	}
	
	public List<KeyCntPair> getSimilarity(String word, int topN) {
		List<KeyCntPair> list = new ArrayList<KeyCntPair>();
		double [] v = this.w2v.get(word);
		for (int i = 0; i < this.allKeyCntPairs.size(); i++) {
			double [] v1 = w2v.get(allKeyCntPairs.get(i).key);
			double cos = MathUtil.cos(v, v1);
			KeyCntPair kcp = new KeyCntPair();
			kcp.key = allKeyCntPairs.get(i).key;
			kcp.value = cos;
			list.add(kcp);
		}
		sort(list);
		return list;
	}

}
