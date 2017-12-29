package deepDriver.dl.aml.lstm.apps.wordSegmentation;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import deepDriver.dl.aml.lstm.IStream;

public class WordSegmentationStream implements IStream, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	WordSegSet wss;	
	int segmentType = 4;
	int vSize = 0;
	int tLength = 40;

	public WordSegmentationStream(WordSegSet wss) {
		super();
		this.wss = wss;
		vSize = wss.cnt;
	}
	
	double[][] sampleTT;
	double[][] targetTT;

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
		return cnt < wss.scentences.size();
	}
	
	Random rd = new Random(System.currentTimeMillis());

	@Override
	public void next() {
		int ri = cnt++;
		double ss = wss.scentences.size(); 
		if (random) {
			ri = (int)(rd.nextDouble() * ss);
		}		 
		next(ri);
	}

	@Override
	public double[][] getSampleTT() {
		return sampleTT;
	}

	@Override
	public double[][] getTarget() {
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

	@Override
	public void next(Object pos) {
		this.pos = (Integer) pos;
		WordSegment ws = wss.scentences.get(this.pos);
		
		List<Integer> sFs = new ArrayList<Integer>();
		List<Integer> tFs = new ArrayList<Integer>();
		while (ws != null) {
			int [] wi = ws.getWordsInt();
			for (int i = 0; i < wi.length; i++) {
				sFs.add(wi[i]);
				tFs.add(getWsType(wi.length, i));
			}
			ws = ws.getNext();
		}
		this.sampleTT = new double[sFs.size()][];
		this.targetTT = new double[tFs.size()][];
		for (int i = 0; i < sampleTT.length; i++) {
			sampleTT[i] = new double[1];
			sampleTT[i][0] = sFs.get(i);
			
			targetTT[i] = new double[segmentType];
			targetTT[i][tFs.get(i)] = 1;
		}
	}
	
	int Ws_B = 0;
	int Ws_M = 1;
	int Ws_E = 2;
	int Ws_S = 3;
	
	public int getWsType(int size, int index) {
		if (size == 1) {
			return Ws_S;
		} else {
			int ti = index + 1;
			if (ti == 1) {
				return Ws_B;
			} else if(ti == size) {
				return Ws_E;
			} else {
				return Ws_M;
			}
		}
		
	}

	@Override
	public IStream[] splitStream(int cnt) { 
		WordSegmentationStream [] iss = new WordSegmentationStream[cnt];
		int slen =(int)((double)(wss.getScentences().size())/(double)cnt);
		List<WordSegment> wslist = wss.getScentences();
		for (int i = 0; i < iss.length; i++) {
			WordSegSet wss1 = new WordSegSet();
			wss1.setCntMap(wss.getCntMap());
			wss1.setIntMap(wss.getIntMap());
			wss1.setStrMap(wss.getStrMap());
			wss1.setCnt(wss.getCnt());
//			List<WordSegment> wslist1 = wss1.getScentences();
			List<WordSegment> al = new ArrayList<WordSegment>();
			int l = slen;
			int s = i * slen;
			if (i == iss.length - 1) {
				l = wss.getScentences().size() - i * slen;
			}
			for (int j = 0; j < l; j++) {
				al.add(wslist.get(s + j));
			}
			wss1.setScentences(al);
			iss[i] = new WordSegmentationStream(wss1);
		}
		
		return iss;
	}

	@Override
	public int splitCnt(int cnt) {
		int slen =(int)((double)(wss.getScentences().size())/(double)cnt);
		return slen;
	}

}
