package deepDriver.dl.aml.string;

public class NFixedStreamImpV2 extends StreamImp {
	int offSet = 0;
	int endPos = 0;
	
	public NFixedStreamImpV2(Dictionary dic, int t, int offSet, int endPos) {
		this(dic, t, offSet);
		this.endPos = endPos;
	}
	
	public NFixedStreamImpV2(Dictionary dic, int t, int offSet) {
		super(dic, t);
		out = false;
		this.offSet = offSet;
		cnt = cnt + offSet;
	}

	public NFixedStreamImpV2(Dictionary dic, int t) {
		super(dic, t);
		out = false;
	}
	
	public boolean hasNext() {		
		if (cnt < 0 || cnt % 2 == 1) {//Q should be 
			cnt = cnt + 1;
		}
		if (endPos > 0) {
			return cnt + groupSize < endPos;
		}
		return cnt + groupSize < dic.txt.size();
	}
	
	@Override
	public void reset() {
		super.reset();
		cnt = cnt + offSet;
	}
	
	int groupSize = 2;
	
	public void next() {
		cnt = cnt + groupSize;
		next(cnt);
	}
	
	public void next(Object pos) {	
		int ri = (Integer) pos;		
		int [] is = dic.txt.get(ri);
		int [] is2 = is;//dic.txt.get(cnt + 1); 
		sampleTT = new double[is.length][];
		targetTT = new double[is.length][];
		StringBuffer sb = null;
		StringBuffer sb2 = null;
		if (out) {
			sb = new StringBuffer();
			sb2 = new StringBuffer();
		}
		
		for (int j = 0; j < sampleTT.length; j++) {	
			int si = 0;
			int ti = 0;
			String endFlag = "b";
			endFlag = dic.EOS;
			if (j > is.length - 1) {
//				si = -1;
				si = dic.strMap.get(dic.EOS);
				if (out) {
					sb.append(dic.EOS);
				}				
/*		} if (j == 0) {
//				si = -1;
				si = dic.strMap.get(dic.EOS);
				if (out) {
					sb.append(dic.EOS);
				}**/			
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
					sb2.append(dic.EOS);
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
		}
		if (out) {
			System.out.println("t:"+sb2.toString());
			System.out.println("s:"+sb.toString());			
		}		
	}
}
