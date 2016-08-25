package deepDriver.dl.aml.lstm.apps.wordSegmentation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

public class WordSegSetV2 extends WordSegSet {
 
	
	public void loadFlatDs(String file) throws IOException {
		System.out.println("Load flat file from "+file);
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
				continue;
			} 
			if (useParser) { 
			} else {
				if (content.length() == 0) {
					content = bi.readLine();
					continue;
				} 
				previous = null;
				//parse the format
				String [] segsFormat = content.split(" ");
				if (segsFormat.length < 2) {
					System.out.println("Invalid formate...");
				}
				for (int k = 0; k < segsFormat.length; k++) {
					WordSegment ws = new WordSegment();
					String seg = segsFormat[k].trim();
					wordNum = wordNum + seg.length();
					if (seg.length() == 0) {
						System.out.println("It is empty....");
						continue;
					}
					wc = wc + seg.length();
				
					string = new int[seg.length()];
					aa = new String[seg.length()];
					for (int i = 0; i < aa.length; i++) {
						aa[i] = seg.substring(i, i + 1);
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
				}
				 
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
	
	public static void main(String[] args) throws IOException {
		WordSegSetV2 wss = new WordSegSetV2();
		wss.setVoLoadOnly(true);		
		wss.loadWordSegSet("D:\\6.workspace\\p.NLP\\train.conll");
		 
		wss.setVoLoadOnly(true);
		wss.loadWordSegSet("D:\\6.workspace\\p.NLP\\dev.conll"); 
		
		wss.setVoLoadOnly(false);		
		wss.setLockVo(true);
		
		String flatFile = "msr_corpus";
		if (args.length > 1) {
			flatFile = args[1];
		}
		wss.loadFlatDs("D:\\6.workspace\\p.NLP\\"+flatFile); 
	}
	
	

}
