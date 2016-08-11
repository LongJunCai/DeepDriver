package deepDriver.dl.aml.distribution.modelParallel;

import java.io.Serializable;

public class ThreadParallel implements Serializable { 
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	class PartialThread extends Thread {
		PartialCallback p;
		int offset;
		int runLen; 	
		
		public PartialThread(PartialCallback p, int offset, int runLen) {
			super();
			this.p = p;
			this.offset = offset;
			this.runLen = runLen;
		}
		
		@Override
		public void run() {
			p.runPartial(offset, runLen);
		}
	};
	
	public void runMutipleThreads(int length, PartialCallback p, int tn) { 
		int eachPart = length/tn;
		PartialThread [] ps = new PartialThread[tn];
		for (int i = 0; i < tn; i++) {
			int offset = i * eachPart;
			int runLen = eachPart;
			if (i == tn - 1) {
				runLen = length - i * eachPart;
			}
			ps[i] = new PartialThread(p, offset, runLen);
			ps[i].start();
		}
		for (int i = 0; i < ps.length; i++) {
			try {
				ps[i].join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

}
