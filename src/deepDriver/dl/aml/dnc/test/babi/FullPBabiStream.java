package deepDriver.dl.aml.dnc.test.babi;

import deepDriver.dl.aml.string.Dictionary;

public class FullPBabiStream extends BabiStream {

	public FullPBabiStream(Dictionary dic, int t) {
		super(dic, t);
	}
	
	public void next(Object pos) {
		if (pa == null) {
			sampleTT = null;
			return;
		} 
		int [] is = pa.getFullTxt();
		int [] a = pa.getFullAnswer();
		tpos = pa.getFullAnswerPos();
		sampleTT = new double[is.length][];
		targetTT = new double[a.length][];
		StringBuffer sb = null;
		StringBuffer sb2 = null;
		if (out) {
			sb = new StringBuffer();
			sb2 = new StringBuffer();
		}
		
		for (int i = 0; i < a.length; i++) {
			int ai = a[i];
			double [] tw = targetTT[i] = new double[targetFeatureNum];
			if (ai >= 1) {
				tw[ai - 1] = 1;
			}
		}	
		
		
		for (int j = 0; j < is.length; j++) {	
			int si = is[j];  
			double [] sw = sampleTT[j] = new double[sampleFeatureNum];			
			if (si >= 1) {
				sw[si - 1] = 1;
			}
			
		}
		if (out) {
			System.out.println("t:"+sb2.toString());
			System.out.println("s:"+sb.toString());			
		}		
		pa = pa.getNext();
	}
	
	int [] tpos;
	
	public int[] getTargetPos() {
		return tpos;
	}

}
