package deepDriver.dl.aml.lstm.apps.wordSegmentation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class WordSegSet {

	List<WordSegment> scentences = new ArrayList<WordSegment>();
	
	Map<String, Integer> strMap = new HashMap<String, Integer>();
	Map<Integer, String> intMap = new HashMap<Integer, String>();
	
	Map<String, Integer> cntMap = new HashMap<String, Integer>();
	boolean useParser = false;
	int wordNum = 0;
	int maxLength = 40;
	
	boolean voLoadOnly = false;
	boolean lockVo = false;
	
	String UnKnown = "*";	

	public boolean isVoLoadOnly() {
		return voLoadOnly;
	}

	public void setVoLoadOnly(boolean voLoadOnly) {
		this.voLoadOnly = voLoadOnly;
	}

	public boolean isLockVo() {
		return lockVo;
	}

	public void setLockVo(boolean lockVo) {
		this.lockVo = lockVo;
	}
	
	int wc = 0;

	public void loadWordSegSet(String file) throws IOException {
		BufferedReader bi = new BufferedReader(new InputStreamReader(
				new FileInputStream(new File(file)), "utf-8"));
		String content = bi.readLine();
		int wordNum = 0;
		
		WordSegment previous = null;
		while (content != null) {
			int[] string = null;
			String[] aa = null;
			content = content.trim();
			if (content.length() == 0) {
				wordNum = 0;
				content = bi.readLine();
				previous = null;
				continue;
			}
			
			
			WordSegment ws = new WordSegment();
			String seg = null;
			if (useParser) {
				aa = parse(content);
				string = new int[aa.length];
				wordNum = wordNum + aa.length;
				if (aa.length > maxLength) {
					System.out.println("l > " + maxLength + ": " + content);
				}
			} else {
//				content = content.toLowerCase();
//				if (content.contains("http")) {
//					int hi = content.indexOf("http");
//					content = content.substring(0, hi);
//				}

//				content = content.replaceAll("[a-z]*", "");
//				content = content.replaceAll("[0-9]*", "");
//				content = content.trim();
				if (content.length() == 0) {
					content = bi.readLine();
					continue;
				}
				
//				if (content.length() > 40) {
//					System.out.println("l > 40: " + content);
//				}
				//parse the format
				String [] segsFormat = content.split("\t");
				if (segsFormat.length < 2) {
					System.out.println("Invalid formate...");
				}
				seg = segsFormat[1].trim();
				wordNum = wordNum + seg.length();
				if (seg.length() == 0) {
					System.out.println("It is empty....");
				}
				wc = wc + seg.length();
//				if (wc >= 338200) {
//					System.out.println("It is a issue. here");
//				}
				//		
//				if (checkCendFlag(seg)) {
//					previous = null;
//				}	
				
				string = new int[seg.length()];
				aa = new String[seg.length()];
				for (int i = 0; i < aa.length; i++) {
					aa[i] = content.substring(i, i + 1);
				}
			}

			for (int i = 0; i < string.length; i++) {
				String s1 = aa[i];								
				if (lockVo) {
					Integer i1 = strMap.get(s1);
					if (i1 == null) {
						i1 = strMap.get(UnKnown);
					}
					string[i] = i1;
				} else {
					string[i] = mapString2Int(s1);
				}				
			}
			
			if (!voLoadOnly) {
				ws.setWords(aa);
				ws.setWordsInt(string);
				ws.setPrevious(previous);
			
				if (previous == null || wordNum >= maxLength) {
					this.scentences.add(ws);//ws is the root;
				} else {
					previous.setNext(ws);
				}
				previous = ws;
			}
			
			if (requireEndFlagCheck && checkCendFlag(seg)) {
				previous = null;
			}	

			content = bi.readLine();
		}
		bi.close();
		System.out.println("There are " + cnt + " words in v");
		
		System.out.println("There are " + wc + " words in all");

		if (requireBlank) {			
			mapString2Int(BLANK);
		}
	}
	public static final String BLANK = "<B>";	
	
	public int getMaxLength() {
		return maxLength;
	}
	
	boolean requireEndFlagCheck = true;	

	public boolean isRequireEndFlagCheck() {
		return requireEndFlagCheck;
	}

	public void setRequireEndFlagCheck(boolean requireEndFlagCheck) {
		this.requireEndFlagCheck = requireEndFlagCheck;
	}

	public void setMaxLength(int maxLength) {
		this.maxLength = maxLength;
	}
	boolean requireBlank = false;
	
	public boolean isRequireBlank() {
		return requireBlank;
	}

	public void setRequireBlank(boolean requireBlank) {
		this.requireBlank = requireBlank;
	}
	String [] cendFlag = {" ", ",", "，", ".", "。", "!", "?", 
			"、", "”","“","《", "》","（", "）", "；"};
	
	public boolean checkCendFlag(String str) {
		for (int i = 0; i < cendFlag.length; i++) {
			if (cendFlag[i].equals(str)) {
				return true;
			}
		}
		return false;
	}
	
	private String[] parse(String content) {
		return null;
	}
	
	public List<WordSegment> getScentences() {
		return scentences;
	}

	public void setScentences(List<WordSegment> scentences) {
		this.scentences = scentences;
	}
	int cnt = 0;

	public int getCnt() {
		return cnt;
	}

	public void setCnt(int cnt) {
		this.cnt = cnt;
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
	
	public void summary() {
		System.out.println("The scentence size is: "+this.scentences.size());
	}
	
	public static void main(String[] args) throws IOException {
		WordSegSet wss = new WordSegSet();
		wss.setMaxLength(1000);
		wss.setRequireBlank(true);
		wss.setRequireEndFlagCheck(false);
		wss.setVoLoadOnly(true); 
		wss.loadWordSegSet("D:\\6.workspace\\p.NLP\\train.conll");
		
		wss.setVoLoadOnly(false); 
		wss.loadWordSegSet("D:\\6.workspace\\p.NLP\\dev.conll");
		wss.summary();
	}

}
