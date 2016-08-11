package deepDriver.dl.aml.string;

public class NFixedStreamImp extends StreamImp {
	int offSet = 0;
	
	public NFixedStreamImp(Dictionary dic, int t, int offSet) {
		super(dic, t);
		out = false;
		this.offSet = offSet;
		cnt = cnt + offSet;
	}

	public NFixedStreamImp(Dictionary dic, int t) {
		super(dic, t);
		out = false;
	}
	
	@Override
	public void reset() {
		super.reset();
		cnt = cnt + offSet;
	}
	
	public void next() {		
		cnt++;
		int [] is = dic.txt.get(cnt);
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
/***/		} else if (j == 0) {
//				si = -1;
				si = dic.strMap.get(dic.EOS);
				if (out) {
					sb.append(dic.EOS);
				}			
			} else {
				si = is[j - 1];
//				si = is[j];
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
			System.out.println("s:"+sb.toString());
			System.out.println("t:"+sb2.toString());
		}		
	}

}
