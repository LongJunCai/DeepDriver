package deepDriver.dl.aml.string;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import deepDriver.dl.aml.lstm.LSTMDataSet;

public class Dictionary implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	int maxCnt = -1;
	int lineNum = 0;
	
	int usageSortedNum = -1;	
	
	public int getUsageSortedNum() {
		return usageSortedNum;
	}

	public void setUsageSortedNum(int usageSortedNum) {
		this.usageSortedNum = usageSortedNum;
	}

	public int getLineNum() {
		return lineNum;
	}

	public void setLineNum(int lineNum) {
		this.lineNum = lineNum;
	}

	public int getMaxCnt() {
		return maxCnt;
	}

	public void setMaxCnt(int maxCnt) {
		this.maxCnt = maxCnt;
	}
	
	class KeyCntPair implements Serializable {
		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;
		String key;
		int cnt;
	}
	
	List<KeyCntPair> keyCntPairs = new ArrayList<KeyCntPair>();
	Map<String, Integer> cntMap = new HashMap<String, Integer>();

	int cnt = 1;
	Map<String, Integer> strMap = new HashMap<String, Integer>();
	Map<Integer, String> intMap = new HashMap<Integer, String>();
	List<int []> txt = new ArrayList<int[]>();
	public static String EOS = "$";
	public static String UNKNOWN = "*";
	public void summaryInf() {
		System.out.println("TXT size is: "+txt.size());
		System.out.println("strMap size is: "+strMap.size());
		System.out.println("intMap size is: "+intMap.size());
	}
	boolean debug = false;	
	
	public boolean isDebug() {
		return debug;
	}

	public void setDebug(boolean debug) {
		this.debug = debug;
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

	public List<int[]> getTxt() {
		return txt;
	}

	public void setTxt(List<int[]> txt) {
		this.txt = txt;
	}

	public int getCnt() {
		return cnt;
	}

	public void setCnt(int cnt) {
		this.cnt = cnt;
	}

	public void loadDicFromFile(String file) throws Exception {
		loadDicFromFile(file, true);
		if (usageSortedNum > 0) {
			sort();
			//
			lineNum = 0;
			cnt = 1;
			summaryInf();
			strMap.clear();
			intMap.clear();
			txt.clear();
			summaryInf();
			//
			remove(usageSortedNum);
			reloadWithSortedDic(usageSortedNum);
			loadDicFromFile(file, false);
			summaryInf();
			if (debug) {
				summaryDic(usageSortedNum);
			}			
		}		
	}
	
	public void reloadWithSortedDic(int num) {
		for (int i = 0; i < num; i++) {
			KeyCntPair kcpi = keyCntPairs.get(i);
			mapString2Int(kcpi.key);
		}
	}
	
	public void remove(int num) {
		for (int i = num; i < keyCntPairs.size(); i++) {
			KeyCntPair kcpi = keyCntPairs.get(i);
			cntMap.remove(kcpi.key);
		}
	}
	int wordNum = 0;
	boolean useParser = false;
	
	public boolean isUseParser() {
		return useParser;
	}

	public void setUseParser(boolean useParser) {
		this.useParser = useParser;
	}
	
	int maxLength = 40;

	public int getMaxLength() {
		return maxLength;
	}

	public void setMaxLength(int maxLength) {
		this.maxLength = maxLength;
	}

	public void loadDicFromFile(String file, boolean loadAll) throws Exception {
		
//		mapString2Int(UNKNOWN);
//		cnt ++;
		BufferedReader bi = new BufferedReader( new InputStreamReader(new 
						FileInputStream(new File(file)), "utf-8"));
		String content = bi.readLine();
		while (content != null) {
			int [] string = null;
			String [] aa = null;
			content = content.trim();
			if (content.length() == 0) {
				content = bi.readLine();
				continue;
			}
			if (useParser) {
				aa = parse(content);	
				string = new int[aa.length];
				wordNum = wordNum + aa.length;
				if (aa.length > maxLength) {
					System.out.println("l > "+maxLength+": "+content);
				}
			} else {
				content = content.toLowerCase();			
				if (content.contains("http")) {
					int hi = content.indexOf("http");
					content = content.substring(0, hi);
				}		
				
				content = content.replaceAll("[a-z]*", "");
				content = content.replaceAll("[0-9]*", "");	
				content = content.trim();
				if (content.length() == 0) {
					content = bi.readLine();
					continue;
				}
				
				wordNum = wordNum + content.length();
				if (content.length() > 40) {
					System.out.println("l > 40: "+content);
				}
				string = new int[content.length() ];
				aa = new String[content.length() ];
				for (int i = 0; i < aa.length; i++) {
					aa[i] = content.substring(i, i + 1);
				}
			}
			
			for (int i = 0; i < string.length; i++) {
				String s1 = EOS;
				if (i < aa.length) {
					s1 = aa[i];
				} else {
					s1 = EOS;
				}
				Integer i1 = strMap.get(s1);
				if (i1 == null) {
					//<a>if it is not in the list, no need
					if (!loadAll) {
						Integer i2 = cntMap.get(s1);
						if (i2 == null) {
							s1 = UNKNOWN;
						}
					}
					//</a>if it is not in the list, no need
					if (maxCnt > 0 && cnt > maxCnt) {
						s1 = UNKNOWN;	
					} 								
				} 
				string[i] = mapString2Int(s1);	
//				else {
//					string[i] = i1.intValue();
//				}
			}
			txt.add(string);
			lineNum ++;
			content = bi.readLine();
		}		
		mapString2Int(EOS);
		bi.close();
		System.out.println("There are "+cnt+" words");
	}
	
	public void summaryDic(int num) {
		for (int i = 0; i < keyCntPairs.size(); i++) {
			KeyCntPair kcpi = keyCntPairs.get(i);
			System.out.println(kcpi.key +strMap.get(kcpi.key));
		}
	}
	
	public void summary() {
		double sum = 0;
		for (int i = 0; i < keyCntPairs.size(); i++) {
			KeyCntPair kcpi = keyCntPairs.get(i);
			sum = sum + kcpi.cnt;
		}
		double sum1 = 0;
		for (int i = 0; i < keyCntPairs.size(); i++) {
			KeyCntPair kcpi = keyCntPairs.get(i);
			sum1 = sum1 + kcpi.cnt;
			System.out.println(i+kcpi.key+kcpi.cnt+":"+sum1/sum);
		}
	}
	
	public void sort() {
		Iterator<String> iter = cntMap.keySet().iterator();
		while (iter.hasNext()) {
			String key = (String) iter.next();
			KeyCntPair kcp = new KeyCntPair();
			kcp.key = key;
			kcp.cnt = cntMap.get(key);			
			keyCntPairs.add(kcp);
		}
		KeyCntPair tkcp = new KeyCntPair();
		for (int i = 0; i < keyCntPairs.size(); i++) {
			KeyCntPair kcpi = keyCntPairs.get(i);
			for (int j = i + 1; j < keyCntPairs.size(); j++) {
				KeyCntPair kcpj = keyCntPairs.get(j);
				if (kcpi.cnt < kcpj.cnt) {
					switchKcp(kcpi, tkcp);
					switchKcp(kcpj, kcpi);
					switchKcp(tkcp, kcpj);
				}
			}
			if (debug) {
				System.out.println(kcpi.key + "," + kcpi.cnt);
			}
//			
		}
	}
	
	public void switchKcp(KeyCntPair s, KeyCntPair t) {
		t.key = s.key;
		t.cnt = s.cnt;
	}
	
	public int mapString2Int(String s1) {
		Integer i1 = strMap.get(s1);
		if (i1 == null) {
			strMap.put(s1, cnt);
			cntMap.put(s1, 1);
			
			intMap.put(cnt, s1);
			return cnt++;				
		} else {
			Integer i2 = cntMap.get(s1);
			cntMap.put(s1, i2 + 1);
			
			return i1.intValue();
		}
	}
	public String decoded(double [][][] targets) {
		return decoded(targets, false);
	}
	
	public String decoded(double [][][] targets, boolean thinData) {
		return decoded(targets, thinData, false);
	}
	
	public String decoded(double [][][] targets, boolean thinData, boolean endWithFlag) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < targets.length; i++) {
			double [][] se = targets[i];
			for (int j = 0; j < se.length; j++) {
				double [] wd = se[j];
				double max = 0;
				int pos = 0;
				if (thinData) {
					pos = (int) wd[0];
				} else {
					for (int k = 0; k < wd.length; k++) {
						if (max < wd[k]) {
							max = wd[k];
							pos = k;
						}
					}
				}
				
				String swd = intMap.get(pos + 1);
				if (swd == null) {
					sb.append("\n");
				} else {
					sb.append(swd);
				}			
				if (endWithFlag && swd.equals(EOS)) {
					break;
				}
			}
//			sb.append("\n");
		}
		return sb.toString();
	}
	public LSTMDataSet encodeSample(String sw, int t) {
		return encodeSample(sw, t, false);
	}
	public LSTMDataSet encodeSample(String sw1, int t, boolean thinData) {
		LSTMDataSet ds = new LSTMDataSet();
		double [][][] samples = new double[1][][];
		ds.setSamples(samples);
		String [] swa = this.parse(sw1);
		samples[0] = new double[t < swa.length ? t : swa.length][];
		double [][] sTT = samples[0];
		for (int i = 0; i < sTT.length; i++) {
			if (thinData) {
				sTT[i] = new double[1];
			} else {
				sTT[i] = new double[cnt];
			}
			
			if (i < swa.length) {
				String wd = swa[i];//.substring(i, i + 1);
				Integer wi = strMap.get(wd);
				if (wi == null) {
					wi = strMap.get(UNKNOWN);					
				}
				if (thinData) {
					sTT[i][0] = wi.intValue() - 1;
				} else {
					sTT[i][wi.intValue() - 1] = 1;
				}
				
			}			
		}
//		samples[0][0] = new double[cnt];
//		int i0 = strMap.get(sw);
//		for (int i = 0; i < samples[0][0].length; i++) {			
//			if ((i + 1) == i0) {
//				samples[0][0][i] = 1;
//			} else {
//				samples[0][0][i] = 0;
//			}		
//		}
		return ds;
	}
	
	public LSTMDataSet encodeSamples(int t) {
		LSTMDataSet ds = new LSTMDataSet();
		double [][][] samples = new double[txt.size()][][];
		double [][][] targets = new double[txt.size()][][];
		ds.setSamples(samples);
		ds.setTargets(targets);
		for (int i = 0; i < samples.length; i++) {
			int [] is = txt.get(i);
			samples[i] = new double[t][];
			targets[i] = new double[t][];
			for (int j = 0; j < samples[i].length; j++) {				
				samples[i][j] = new double[cnt];
				targets[i][j] = new double[cnt];
				if (j > is.length - 2) {
					continue;
				}
				int si = is[j];
				int ti = is[j + 1];
				double [] sw = samples[i][j];
				double [] tw = targets[i][j];
				for (int j2 = 0; j2 < sw.length; j2++) {
					if ((j2 + 1) == si) {
						sw[j2] = 1;
					} else {
						sw[j2] = 0;
					}
					if ((j2 + 1) == ti) {
						tw[j2] = 1;
					} else {
						tw[j2] = 0;
					}
				}
			}
		}
		return ds;
	}
	
	public static void main2(String[] args) throws Exception {
		Dictionary dic = new Dictionary();
		dic.setUseParser(true);
		dic.setDebug(true);
		dic.setUsageSortedNum(1000);
		dic.loadDicFromFile("D:\\6.workspace\\ANN\\lstm\\talk2015_2016.txt");
		System.out.println(dic.cnt);
		System.out.println("lines: "+dic.lineNum);
		System.out.println("words: "+dic.wordNum);
		System.out.println("words per line: "+(double)dic.wordNum/(double)dic.lineNum);
		
		System.out.println(dic.strMap.get(dic.EOS));
		System.out.println(dic.strMap.get(dic.UNKNOWN));
		String start = "我真的需要你";
		int t = 5;
//		String start = "我";
		LSTMDataSet qds = dic.encodeSample(start, start.length());
		LSTMDataSet ads = dic.encodeSample("X", t);
		dic.summaryDic(1000);
	}
	
	String [] urlEndFlag = {" ", ",", "，"};
	String [] cendFlag = {" ", ",", "，", ".", "。", "!", "?"};
	public int getFirstEndFlag(String str, String [] endFlag) {
		int epos = 0;
		boolean hit = false;
		for (int i = 0; i < endFlag.length; i++) {
			int a = str.indexOf(endFlag[i]);
			if (a >= 0 && (!hit || epos > a)) {
				hit = true;
				epos = a;
			}
		}
		if (!hit) {
			epos = -1;
		}
		return epos;
	}
	
	public String segmentWord(String str, List<String> al, int epos) {
		if (epos < 0) {
			al.add(str);
			str = "";
		} else {
			String a = str.substring(0, epos);
			al.add(a);
			str = str.substring(epos);
		}
		return str;
	}
	
	public boolean contains(String [] aa, String a) {
		for (int i = 0; i < aa.length; i++) {
			if (aa[i].equals(a)) {
				return true;
			}
		}
		return false;
	}
	
	public int getFirstNonDiEn(String s) {
		for (int i = 0; i < s.length(); i++) {			
			String a = s.substring(i, i+1);
			if (a.matches("[a-z]") || a.matches("[0-9]")) {			
				continue;
			} else {
				return i;
			}			
		}
		return -1;
	}
	
	public int getLastEndFlag(List<String> al, String [] aa) {
		for (int i = al.size() - 1; i >= 0; i-- ) {
			String a = al.get(i);
			for (int j = 0; j < aa.length; j++) {
				if (a.equals(aa[j])) {
					return i;
				}
			}
		}
		return -1;
	}
	
	public String [] parse(String str) {
		str = str.toLowerCase();
		List<String> al = new ArrayList<String>();
		String [] aa = null;
		int en = 1;
		int flag = 2;
		int normal = 3;
		int lastFlag = normal; 
		while (str.length() > 0) {		
			if (al.size() >= maxLength) {
				int p = getLastEndFlag(al, cendFlag);
				if (p > maxLength/2) {
					for (int i = al.size() - 1; i > p; i--) {
						al.remove(i);
					}
				}
				break;
			}
			if (str.startsWith("http")) {
				int epos = getFirstEndFlag(str, urlEndFlag);					
				str = segmentWord(str, al, epos);
				lastFlag = normal;
				continue;
			} 
			String a = str.substring(0, 1);
			if (a.matches("[a-z]") || a.matches("[0-9]")) {
				int epos = getFirstNonDiEn(str);	
				str = segmentWord(str, al, epos);	
				lastFlag = en;				
				continue;
			}
			if (lastFlag == flag) {
				if (!contains(cendFlag, a)) {
					al.add(a);
				}				
			} else {
				al.add(a);
			}
			if (contains(cendFlag, a)) {
				lastFlag = flag;
			} else {
				lastFlag = normal;
			}
			str = str.substring(1);			
		}
		aa = new String[al.size()];
		for (int i = 0; i < aa.length; i++) {
			aa[i] = al.get(i);
		}
		return aa;
	}
	
	public static void main(String[] args) {
		String a = "abc,123, 我aa真的123需要你 hello world, http://www.iqiyi.com/v_19rro1nweo.html 我真的需要你VIP，需要你";
		Dictionary dic = new Dictionary();
		String [] aa = dic.parse(a);
		System.out.println(a);
		for (int i = 0; i < aa.length; i++) {
			System.out.println(aa[i]);
		}		
	}

}
