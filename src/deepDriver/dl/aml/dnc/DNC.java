package deepDriver.dl.aml.dnc;

import deepDriver.dl.aml.lstm.IStream;
import deepDriver.dl.aml.math.MathUtil;

public class DNC {
	
	DNCConfigurator cfg;
	
	DNCBPTT bptt;
	
	
	
	public DNC(DNCConfigurator cfg) {
		super();
		this.cfg = cfg;
		bptt = new DNCBPTT(cfg);
	}
	
	 
	int maxSteps = 0;

	public void train(ITxtStream is) { 		
		int allCnt = 0;
		double lastAvgErr = 0;
		for (int i = 0; i < cfg.trainingLoop; i++) {
			int cnt = 0;
			int correctCnt = 0;
			double err = 0;
			is.reset();			
			while (is.hasNext()) {
				is.next();
				double [][] x = is.getSampleTT();
				if (x == null) {
					System.out.println("Finished "+i+" epochs.");
					break;
				}
				bptt.setxSteps(x.length);
				if (maxSteps < x.length) {
					maxSteps = x.length;
				}
				int [] pos = is.getTargetPos();
				double [][] result = new double[x.length][];
				for (int j = 0; j < x.length; j++) {
					result[j] = bptt.forward(x[j]);
				}				
				
				if (pos == null) {
					if (is.getTarget() != null) {
						cnt ++;
						allCnt ++;
						double [] ta = is.getTarget()[0];					
						bptt.backWard(is.getTarget(), pos);
						err = err + bptt.getError();
						bptt.updateWws();
						if (MathUtil.check(result[result.length - 1], ta)) {
							correctCnt ++;
						} 
					}
				} else {
					bptt.backWard(is.getTarget(), pos);
					bptt.updateWws();
					err = err + bptt.getError();					
					for (int j = 0; j < pos.length; j++) {
						cnt ++;
						allCnt ++;
						if (MathUtil.check(result[pos[j]], is.getTarget()[j])) {
							correctCnt ++;
						}
					}
				}
				
					
				if (cnt % 10 == 0) {
					System.out.println(allCnt+" are tested, and "+cnt+" in this epoch," +
							"error is "+err/(double)cnt+" ,and "+(double)correctCnt/(double)cnt);
				}
			}
			
			if (allCnt > 0) {
					double avgErr = err/(double)cnt;
					
					if (lastAvgErr == 0) {						 
					} else if (allCnt % cfg.ldecayLoop == 0 || lastAvgErr < avgErr) {
//					} else if (allCnt % 20000 == 0) {
						if (cfg.l/10.0 >= cfg.ml) {
							cfg.l = cfg.l/10.0;
						}
					}
					lastAvgErr = avgErr;
			}
			
			System.out.println(allCnt+" are tested, and "+cnt+" in this epoch," +
					"error is "+err/(double)cnt+" ,and "+(double)correctCnt/(double)cnt);
				
			System.out.println("Max Steps "+maxSteps);
			System.out.println("The learning rate is: "+cfg.l);
		}
	}
	
	
	public void test(IStream is) { 
		for (int i = 0; i < cfg.trainingLoop; i++) {
			is.reset();
			while (is.hasNext()) {
				is.next();
				double [][] x = is.getSampleTT();
				bptt.forward(x[0]); 
				if (is.getTarget() != null) {
					bptt.reset();
				}
			}
		}
	}
	
	
}
