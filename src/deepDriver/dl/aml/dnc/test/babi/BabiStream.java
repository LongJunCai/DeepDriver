package deepDriver.dl.aml.dnc.test.babi;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import deepDriver.dl.aml.dnc.ITxtStream;
import deepDriver.dl.aml.lstm.IStream;
import deepDriver.dl.aml.random.RandomFactory;
import deepDriver.dl.aml.string.Dictionary;

public class BabiStream implements ITxtStream {
	int tTLength;
	int sampleFeatureNum;
	int targetFeatureNum;
	Dictionary dic;	
	
	Map<String, Integer> strEnMap = new HashMap<String, Integer>();
	Map<Integer, String> intEnMap = new HashMap<Integer, String>();
	
	public BabiStream(Dictionary dic, int t) {
		this.tTLength = t;
		this.dic = dic;
		sampleFeatureNum = dic.getCnt();
		targetFeatureNum = dic.getCnt();
		sampleTT = new double[t][];
		targetTT = new double[t][];
		for (int i = 0; i < sampleTT.length; i++) {
			sampleTT[i] = new double[sampleFeatureNum];
			targetTT[i] = new double[targetFeatureNum];
		}
		
		INT_SPACE = dic.getStrMap().get(SPACE);
		INT_TAB = dic.getStrMap().get(TAB);		
		INT_NUM_1 = dic.getStrMap().get(NUM_1);
		INT_QUESTION = dic.getStrMap().get(QUESTION);
		constructParagraph();
	} 
	
	public int getSampleTTLength() {
		return tTLength;
	}
	public int getSampleFeatureNum() {
		return sampleFeatureNum;
	}
	public int getTargetFeatureNum() {
		return targetFeatureNum;
	}
	
	int cnt = -1;
	double[][] sampleTT;
	double[][] targetTT;
	
	public boolean hasNext() {		
		return true;
	}
	
	public void reset() {
		cnt = -1;
		pa = root;
		Paragraph pa1 = root;
		pa1.reset();
		while (pa1.next != null) {
			pa1 = pa1.next;
			pa1.reset();			
		}
	}
	 
	
	public void next() {
		next(cnt);
	}
	boolean out = false;
	
	static String NUM_1 = "1";
	static int INT_NUM_1 = -1;
	static String SPACE = " ";
	static int INT_SPACE = -1;
	
	static String TAB = "\t";
	static int INT_TAB = -1;
	
	static String QUESTION = "?";
	static int INT_QUESTION = -1;
	
	Paragraph root;
	Paragraph pa = null;
	
	List<Paragraph> pas = new ArrayList<>();
	boolean shuffle = false;
	
	public void constructParagraph() {
		cnt ++; 
		int [] is = dic.getTxt().get(cnt);
		
		while (true) {
			if (is[0] == INT_NUM_1) {
				Paragraph pa1 = new Paragraph();
				pas.add(pa1);
				
				pa1.intEnMap = intEnMap;
				if (pa != null) {
					pa.next = pa1;
				}
				pa = pa1;
				if (root == null) {
					root = pa;
				}
			}
			
			boolean qm = false;			
			for (int i = 1; i < is.length; i++) { 
				if (is[i] != INT_SPACE && is[i] != INT_TAB) {
					if (!qm) {
						String str = dic.getIntMap().get(is[i]);
						pa.addWord(Str2Int(str));
						if (is[i] == INT_QUESTION) {
							qm = true;
						} 
					} else {
						String str = dic.getIntMap().get(is[i]);
						pa.addAnswer(Str2Int(str));
						break;
					}
				}				
			}
			
			cnt ++; 
			if (cnt > dic.getTxt().size() - 1) {
				System.out.println("Finished the paragraph construction, there are "+enCnt+" words left.");
				sampleFeatureNum = enCnt;
				targetFeatureNum = enCnt;
				pa = root;
				break;
			}
			is = dic.getTxt().get(cnt);
		}
	}
	
	int enCnt = 0;
	public int Str2Int(String s) {
		Integer int1 = strEnMap.get(s);
		if (int1 == null) {			
			int1 = ++enCnt;
			strEnMap.put(s, int1);
			intEnMap.put(int1, s);
		}
		return int1;
	}
	
	public static Random random = RandomFactory.getRandom();
	private Paragraph getPara() {
		if (shuffle) {
			cnt ++;
			int size = pas.size();
			if (cnt >= size) {
				return null;
			}
			int index = (int)((double)size * random.nextDouble());
			Paragraph pa1 = pas.get(index);
			pa1.reset();
			return pa1; 
		} else {
			pa = pa.next;
			return pa;
		}		
	}
	
	public void next(Object pos) {
		int [] is = pa.nextTxt();		
		if (is == null) {
			pa = getPara();
			if (pa == null) {
				sampleTT = null;
				return;
			} else {
				is = pa.nextTxt(); 
			}
		}
		int a = pa.getAnswer();
		sampleTT = new double[is.length][];
		targetTT = new double[1][];
		StringBuffer sb = null;
		StringBuffer sb2 = null;
		if (out) {
			sb = new StringBuffer();
			sb2 = new StringBuffer();
		}
		
		double [] tw = targetTT[0] = new double[targetFeatureNum];
		if (a >= 1) {
			tw[a - 1] = 1;
		}
		
		for (int j = 0; j < is.length; j++) {	
			int si = 0;
			si = is[j];  
			double [] sw = sampleTT[j] = new double[sampleFeatureNum];			
			if (si >= 1) {
				sw[si - 1] = 1;
			}
			
		}
		if (out) {
			System.out.println("t:"+sb2.toString());
			System.out.println("s:"+sb.toString());			
		}		
	}

	public double[][] getSampleTT() {
		return sampleTT;
	}
	@Override
	public double[][] getTarget() {
		return targetTT;
	} 

	@Override
	public Object getPos() {
		return cnt;
	}

	@Override
	public int[] getTargetPos() {
		return null;
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
