package deepDriver.dl.aml.w2v;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import deepDriver.dl.aml.lstm.apps.wordSegmentation.WordSegSet;
import deepDriver.dl.aml.lstm.apps.wordSegmentation.WordSegment;
import deepDriver.dl.aml.stream.IWordStream;

public class Window4WordSegStream implements IWordStream {
	
	WordSegSet wss;	
	int segmentType = 4;
	int vSize = 0;
	int tLength = 40;
	
	int cxtLength = 5;

	public Window4WordSegStream(WordSegSet wss) {
		super();
		this.wss = wss;
		vSize = wss.getCnt();
	}
	
	String [] sampleTT;
	String [] targetTT;

	@Override
	public void reset() {
		cnt = 0;
	}
	
	int cnt = 0;
	boolean random = true;	

	public boolean isRandom() {
		return random;
	}

	public void setRandom(boolean random) {
		this.random = random;
	}

	@Override
	public boolean hasNext() {
		if (!finishScentence) {
			return true;
		}
		return cnt < wss.getScentences().size();
	}
	
	Random rd = new Random(System.currentTimeMillis());

	@Override
	public void next() {
		int ri = cnt;
		if (finishScentence) {
			cnt++;
		}
//		double ss = wss.getScentences().size(); 
//		if (random) {
//			ri = (int)(rd.nextDouble() * ss);
//		}		 
		next(ri);
	}

	@Override
	public String [] getSampleTT() {
		return sampleTT;
	}

	@Override
	public String [] getTarget() {
		return targetTT;
	}

	@Override
	public int getSampleTTLength() {
		return tLength;
	}

	@Override
	public int getSampleFeatureNum() {
		return vSize;
	}

	@Override
	public int getTargetFeatureNum() {
		return segmentType;
	}
	
	int pos;

	@Override
	public Object getPos() {
		return pos;
	}
	
	boolean finishScentence = true;
	int posOfSen = 0;
	List<String> sFs = new ArrayList<String>();
	
	public String arrary2String(String [] a) {//the JVM optimize the stringbuffer for this.
		String s = "";
		for (int i = 0; i < a.length; i++) {
			s = s + a[i];
		}
		return s;
	}
	
	@Override
	public void next(Object pos) {
		this.pos = (Integer) pos;
		if (finishScentence) {
			sFs.clear();
			posOfSen = 0;
			WordSegment ws = wss.getScentences().get(this.pos);		
			while (ws != null) {
				String wi = arrary2String(ws.getWords());
				sFs.add(wi);
				ws = ws.getNext();
			}
			finishScentence = false;
		}
		
		this.sampleTT = new String[cxtLength * 2];
		this.targetTT = new String[1];
		targetTT[0] = sFs.get(posOfSen);
		
		for (int i = 0; i < sampleTT.length; i++) {
			int j = i;
			if (i >= cxtLength) {
				j = i + 1;
			}
			int mi = j - cxtLength + posOfSen; 
			if (mi < 0 || mi > sFs.size() - 1) {
				sampleTT[i] = wss.BLANK;
			} else {
				sampleTT[i] = sFs.get(mi);
			}					
		}
		if (posOfSen == sFs.size() - 1) {
			finishScentence = true;
		}
		posOfSen ++;		
	}
	
	

}
