package deepDriver.dl.aml.string;

import deepDriver.dl.aml.lstm.IStream;

public class StreamImp implements IStream {
	
	int tTLength;
	int sampleFeatureNum;
	int targetFeatureNum;
	Dictionary dic;	
	
	public StreamImp(Dictionary dic, int t) {
		this.tTLength = t;
		this.dic = dic;
		sampleFeatureNum = dic.cnt;
		targetFeatureNum = dic.cnt;
		sampleTT = new double[t][];
		targetTT = new double[t][];
		for (int i = 0; i < sampleTT.length; i++) {
			sampleTT[i] = new double[sampleFeatureNum];
			targetTT[i] = new double[targetFeatureNum];
		}
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
	@Override
	public boolean hasNext() {
		return cnt + 2 < dic.txt.size();
	}
	boolean out = true;
	@Override
	public void next() {		
		cnt++;
		int [] is = dic.txt.get(cnt);
		int [] is2 = is;//dic.txt.get(cnt + 1); 
		StringBuffer sb = null;
		StringBuffer sb2 = null;
		if (out) {
			sb = new StringBuffer();
			sb2 = new StringBuffer();
		}
		
		for (int j = 0; j < sampleTT.length; j++) {	
			int si = 0;
			int ti = 0;
			if (j > is.length - 1) {
//				si = -1;
				si = dic.strMap.get(dic.EOS);
				if (out) {
					sb.append("b");
				}				
			} else {
				si = is[j];
				if (out) {
					sb.append(dic.intMap.get(si));
				}				
			}	
			if (j + 1 > is2.length - 1) {
//				ti = -1;
				ti = dic.strMap.get(dic.EOS);
				if (out) {
					sb2.append("b");
				}				
			} else {
				ti = is2[j + 1];
				if (out) {
					sb2.append(dic.intMap.get(ti));
				}				
			}
			double [] sw = sampleTT[j] = new double[sampleFeatureNum];
			double [] tw = targetTT[j] = new double[targetFeatureNum];
			if (si >= 1) {
				sw[si - 1] = 1;
			}
			if (ti >= 1) {
				tw[ti - 1] = 1;
			}
			
//			double [] sw = sampleTT[j];
//			double [] tw = targetTT[j];
//			for (int j2 = 0; j2 < sw.length; j2++) {
//				if ((j2 + 1) == si) {
//					sw[j2] = 1;
//				} else {
//					sw[j2] = 0;
//				}
//				if ((j2 + 1) == ti) {
//					tw[j2] = 1;
//				} else {
//					tw[j2] = 0;
//				}
//			}
		}
		if (out) {
			System.out.println("s:"+sb.toString());
			System.out.println("t:"+sb2.toString());
		}		
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
	public void reset() {
		cnt = -1;
	}


	@Override
	public Object getPos() {
		return cnt;
	}


	@Override
	public void next(Object pos) {
		cnt = (Integer) pos;
	}
		

}
