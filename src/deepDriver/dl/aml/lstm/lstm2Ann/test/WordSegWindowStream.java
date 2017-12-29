package deepDriver.dl.aml.lstm.lstm2Ann.test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import deepDriver.dl.aml.lstm.IStream;
import deepDriver.dl.aml.lstm.apps.wordSegmentation.WordSegSet;
import deepDriver.dl.aml.lstm.apps.wordSegmentation.WordSegment;

public class WordSegWindowStream implements IStream {
	
	WordSegSet wss;	
	int segmentType = 4;
	int vSize = 0;
	int tLength = 40;
	
	int cxtLength = 3;

	public WordSegWindowStream(WordSegSet wss) {
		super();
		this.wss = wss;
		vSize = wss.getCnt();
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
	
	boolean finishScentence = true;
	int posOfSen = 0;
	List<Integer> sFs = new ArrayList<Integer>();
	List<Integer> tFs = new ArrayList<Integer>();
	
	@Override
	public void next(Object pos) {
		this.pos = (Integer) pos;
		if (finishScentence) {
			sFs.clear();
			tFs.clear();
			posOfSen = 0;
			WordSegment ws = wss.getScentences().get(this.pos);		
			while (ws != null) {
				int[] wi = ws.getWordsInt();
				for (int i = 0; i < wi.length; i++) {
					sFs.add(wi[i]);
					tFs.add(getWsType(wi.length, i));
				}
				ws = ws.getNext();
			}
			finishScentence = false;
		}
		
		this.sampleTT = new double[cxtLength * 2 + 1][];
		this.targetTT = new double[1][];
		targetTT[0] = new double[segmentType];
		if (posOfSen > tFs.size() - 1) {
			System.out.println("what is wrong.");
		}
		int ttype = tFs.get(posOfSen);
		targetTT[0][ttype] = 1;
		for (int i = 0; i < sampleTT.length; i++) {
			int mi = i - cxtLength + posOfSen;
			sampleTT[i] = new double[1];
			if (mi < 0 || mi > sFs.size() - 1) {
				sampleTT[i][0] = wss.mapString2Int(wss.BLANK);
			} else {
				sampleTT[i][0] = sFs.get(mi);
			}					
		}
		if (posOfSen == sFs.size() - 1) {
			finishScentence = true;
		}
		posOfSen ++;		
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
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int splitCnt(int cnt) {
		// TODO Auto-generated method stub
		return 0;
	}

}
