package deepDriver.dl.aml.w2v;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import deepDriver.dl.aml.lstm.apps.wordSegmentation.WordSegSet;
import deepDriver.dl.aml.stream.IWordStream;

public class NativeWordStream implements IWordStream {
	
	String file;
	int cxtLength = 5;
	boolean finishScentence = true;
	int posOfSen = 0;

	public NativeWordStream(String file) {
		super();
		this.file = file;
	}
	
	BufferedReader bi = null;
	int vSize = 0;
	int tLength = 40;
	String [] sampleTT;
	String [] targetTT;

	int cnt;
	@Override
	public void reset() {
		try {
			if (bi != null) {
				bi.close();
			}
			bi = new BufferedReader(new InputStreamReader(
					new FileInputStream(new File(file)), "utf-8"));
		} catch (Exception e) { 
			e.printStackTrace();
		}
	}

	String content;
	@Override
	public boolean hasNext() {
		if (finishScentence) {
			try {
				content = bi.readLine();
				if (content == null) {
					return false;
				}
				content = content.trim();
				while (content.length() == 0) {
					content = bi.readLine();
					if (content == null) {
						return false;
					}
					content = content.trim();
				}
				cnt ++;
				
			} catch (IOException e) { 
				e.printStackTrace();
			}
		}
		return true;
	}

	@Override
	public void next() {
		next(this.posOfSen);
	}

	@Override
	public String[] getSampleTT() {
 		return sampleTT;
	}

	@Override
	public String[] getTarget() {
 		return targetTT;
	}

	@Override
	public int getSampleTTLength() {
 		return vSize;
	}

	@Override
	public int getSampleFeatureNum() { 
		return tLength;
	}

	@Override
	public int getTargetFeatureNum() { 
		return 0;
	}

	@Override
	public Object getPos() { 
		return cnt;
	}

	@Override
	public void next(Object pos) { 
		String [] segsFormat = content.split(" ");
		this.sampleTT = new String[cxtLength * 2];
		this.targetTT = new String[1];
		targetTT[0] = segsFormat[posOfSen];
		for (int i = 0; i < sampleTT.length; i++) {
			int j = i;
			if (i >= cxtLength) {
				j = i + 1;
			}
			int mi = j - cxtLength + posOfSen; 
			if (mi < 0 || mi > segsFormat.length - 1) {
				sampleTT[i] = WordSegSet.BLANK;
			} else {
				sampleTT[i] = segsFormat[mi];
			}					
		}
		posOfSen++;
		if (posOfSen == segsFormat.length) {
			posOfSen = 0;
			finishScentence = true;
		} else {
			finishScentence = false;
		}
	}

}
