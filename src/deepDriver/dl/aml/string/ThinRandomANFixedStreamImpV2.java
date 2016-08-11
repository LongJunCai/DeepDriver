package deepDriver.dl.aml.string;

import java.util.Random;

public class ThinRandomANFixedStreamImpV2 extends ANFixedStreamImpV2 {
	int offSet = 0;
	int endPos = 0;
	
	public ThinRandomANFixedStreamImpV2(Dictionary dic, int t, int offSet, int endPos, long time) {
		this(dic, t, offSet);
		this.endPos = endPos;
		rd = new Random(time);
	}
	public ThinRandomANFixedStreamImpV2(Dictionary dic, int t, int offSet, long time) {
		super(dic, t);
		out = false;
		this.offSet = offSet;
		cnt = cnt + offSet;
		rd = new Random(time);
	}

	public ThinRandomANFixedStreamImpV2(Dictionary dic, int t, long time) {
		super(dic, t);
		out = false;
		rd = new Random(time);
	}
	
	public boolean hasNext() {
//		if (cnt % 2 == 0) {//Q should be 
//			cnt = cnt + 1;
//		}
//		if (cnt < 0 || cnt % 2 == 0) {//A should be 
//			cnt = cnt + 1;
//		}
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
	Random rd;
	
	public Object getPos() {
		return currPos;
	}

	int currPos;
	public void next() {		
		double l = (dic.txt.size())/groupSize;
		int ri = (int)(l * rd.nextDouble()) * groupSize;
		cnt = cnt + groupSize;
		currPos = ri;
		next(ri);
	}
	
	public void next(Object pos) {		
//		double l = (dic.txt.size())/groupSize;
//		int ri = (int)(l * rd.nextDouble()) * groupSize;
//		cnt = cnt + groupSize;
		int ri = (Integer) pos;
		int [] is = dic.txt.get(ri + 1);
		int [] is2 = is;//dic.txt.get(cnt + 1); 
		sampleTT = new double[is.length + 1][];
		targetTT = new double[is.length + 1][];
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
			if (j -1 > is.length - 1) {
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
				si = is[j -1];
				if (out) {
					sb.append(dic.intMap.get(si));
				}				
			}	
			if (j > is2.length - 1) {
//				ti = -1;
				ti = dic.strMap.get(dic.EOS);
				if (out) {
					sb2.append(dic.EOS);
				}				
			} else {
				ti = is2[j];
				if (out) {
					sb2.append(dic.intMap.get(ti));
				}				
			}
			double [] sw = sampleTT[j] = new double[1];
			double [] tw = targetTT[j] = new double[targetFeatureNum];
			if (si >= 1) {
				sw[0] = si - 1;//thin one.
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
