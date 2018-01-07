package deepDriver.dl.aml.dnc;

import deepDriver.dl.aml.math.MathUtil;


public class DNCBPTT {
	
	DNCConfigurator cfg;
	int t;
	
	public DNCBPTT(DNCConfigurator cfg) {
		super();
		this.cfg = cfg;
	}
	
	boolean init = false;
	public void reset() {
		init = false;
	}
	
	int xSteps = 0; 
	
	public int getxSteps() {
		return xSteps;
	} 
	
	public void setxSteps(int xSteps) {
		this.xSteps = xSteps;
	}
	
	public void prepareEnv() {
		cfg.controller.prepareEnv();
		cfg.memory.prepareEnv();
		cfg.writeHead.prepareEnv(); 
		for (int i = 0; i < cfg.readHeads.length; i++) {
			cfg.readHeads[i].prepareEnv();
		}
	}
	
	long ci = 0;
	long pi = 0;
	long gp = 0;
	long am = 0;
	
	long wa = 0;
	long wl = 0;
	
	long wm = 0;
	long ra = 0;
	long rm = 0;
	
	long mg = 0;
	long ot = 0;
	
	public void summary() {
		double all = ci + pi + gp + am + wa + wl + wm + ra + rm + mg + ot;
		System.out.println("ci"+(double)ci/all + "pi"+(double)pi/all + "gp"+(double)gp/all
				 + "am"+(double)am/all  + "wa"+(double)wa/all + "wl"+(double)wl/all
				 + "wm"+(double)wm/all + "ra"+(double)ra/all + "rm"+(double)rm/all
				 + "mg"+(double)mg/all + "ot"+(double)ot/all);

	}
	
	int fcnt = 0;
	public double [] forward(double [] x, int [] pos, int length) {
		this.cfg.controller.dbptt = this;
		this.cfg.memory.setBptt(this);
		DNCReadHead [] rh = cfg.readHeads;
		for (int i = 0; i < rh.length; i++) {
			rh[i].bptt = this;
		}
		cfg.writeHead.bptt = this;
		if (!init) {
			init = true;
			t = 0;
			fcnt = 0;
		} else {
			t ++;
		}
		
		double [] result = null;		
		long t1 = System.currentTimeMillis();
		//(1) construct input
		double [] xt = constructInput(x);
		ci = ci + System.currentTimeMillis() - t1;
		t1 = System.currentTimeMillis();
		
		//(1.1) process input by controller
		double [] hts = processInput(xt);//how to ensure the xt's length fits controller.
		pi = pi + System.currentTimeMillis() - t1;
		t1 = System.currentTimeMillis();
		
		//(2) generate the interface params;
		generateInterfaceParameters(hts);
		gp = gp + System.currentTimeMillis() - t1;
		t1 = System.currentTimeMillis();
		
		//(3) allocate memory usage
		allocateMemory();
		am = am + System.currentTimeMillis() - t1;
		t1 = System.currentTimeMillis();
		
		//(3.1) write memory addressing
		writeHeadAddressing();
		wa = wa + System.currentTimeMillis() - t1;
		t1 = System.currentTimeMillis();
		
		//(3.2) memory writing linkage
		writeMemoryLinkage();
		wl = wl + System.currentTimeMillis() - t1;
		t1 = System.currentTimeMillis();
		
		//(3.3) write memory
		writeMemory();
		wm = wm + System.currentTimeMillis() - t1;
		t1 = System.currentTimeMillis();
		
		//(3.4) read head addressing
		readHeadAddressing();
		ra = ra + System.currentTimeMillis() - t1;
		t1 = System.currentTimeMillis();
		
		//(4) read from memory
		readMemory();
		rm = rm + System.currentTimeMillis() - t1;
		t1 = System.currentTimeMillis();
		
		//(5) merge result from controller and read heads.
		mergeResult();
		mg = mg + System.currentTimeMillis() - t1;
		t1 = System.currentTimeMillis();
		
		//(6) regression or classify the results.	
		if (pos == null) {
			if (t == length - 1) { 
				result = output(length, pos);
			} else {
				result = null;
			}			
		} else {
			if (t == pos[fcnt]) { 				
				result = output(length, pos);
				fcnt ++;
			}
		}
		ot = ot + System.currentTimeMillis() - t1;
		t1 = System.currentTimeMillis();
			
		return result;
	}
	
	
	
	
	public double [] output(int length, int [] pos) {
		return cfg.controller.output(length, pos);
	}
	
	public void mergeResult() {
		cfg.controller.mergeResult();		
	}
	
	public void readMemory2() {
		for (int i = 0; i < this.cfg.readHeads.length; i++) {
			cfg.readHeads[i].readMemory();
		}
	} 
	
	public void readMemory() {
		Thread [] ts = new Thread[cfg.readHeads.length];
		for (int i = 0; i < cfg.readHeads.length; i++) {
			final int id = i;
			ts[i] = new Thread() {
				public void run() {
					cfg.readHeads[id].readMemory();
				}				
			};
			ts[i].start();
		} 
		
		try {
			for (int i = 0; i < ts.length; i++) {
				ts[i].join();
			} 
		} catch (Exception e) {
			e.printStackTrace();
		}
	} 
	
	public void writeMemoryLinkage() {
		cfg.memory.writeMemoryLinkage();
	}
	
	public void writeMemory() { 
		cfg.writeHead.writeMemory();
	}
	
	public void writeHeadAddressing() {
		cfg.writeHead.addressing();
	}
	
	public void readHeadAddressing() {
		Thread [] ts = new Thread[cfg.readHeads.length];
		for (int i = 0; i < cfg.readHeads.length; i++) {
			final int id = i;
			ts[i] = new Thread() {
				public void run() {
					cfg.readHeads[id].readHeadAddressing();
				}				
			};
			ts[i].start();
		} 
		
		try {
			for (int i = 0; i < ts.length; i++) {
				ts[i].join();
			} 
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void readHeadAddressing2() {
		for (int i = 0; i < cfg.readHeads.length; i++) {
			cfg.readHeads[i].readHeadAddressing();
		} 
	}
	
	public void allocateMemory() {
		cfg.memory.allocateMemory();
	}
	
	public double[] constructInput(double [] x) {
		return cfg.controller.constructInput(x);
	}
	
	public double [] processInput(double [] x) {
		return cfg.controller.processInput(new double [][] {x}); //one step each time.
	}
	
	public void generateInterfaceParameters(double [] hts) {
		for (int i = 0; i < cfg.readHeads.length; i++) {
			cfg.readHeads[i].generateInterfaceParameters(hts);
		}
		cfg.writeHead.generateInterfaceParameters(hts);
	}
	
	public void generateInterfaceParameters3(final double [] hts) {
		Thread [] ts = new Thread[cfg.readHeads.length];
		for (int i = 0; i < cfg.readHeads.length; i++) {
			final int id = i;
			ts[i] = new Thread() {
				public void run() {
					cfg.readHeads[id].generateInterfaceParameters(hts);
				}				
			};
			ts[i].start();
		}
		Thread wt = new Thread() {
			public void run() {
				cfg.writeHead.generateInterfaceParameters(hts);
			}
		};
		wt.start();
		
		try {
			for (int i = 0; i < ts.length; i++) {
				ts[i].join();
			}
			wt.join();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	int lastT;
	
	public void updateWws() {
		updateConstructInput();
		//(1.1) process input by controller
		updateProcessInput();//how to ensure the xt's length fits controller.
		//(2) generate the interface params;
		updateGenerateInterfaceParameters();
		//(3) allocate memory usage
		updateAllocateMemory();
		//(3.1) write memory addressing
		updateWriteHeadAddressing();		
		//(3.2) memory writing linkage
		updateWriteMemoryLinkage();
		//(3.3) write memory
		updateWriteMemory();
		//(3.4) read head addressing
		updateReadHeadAddressing();
		//(4) read from memory
		updateReadMemory();
		//(5) merge result from controller and read heads.
		updateMergeResult();
		//(6) regression or classify the results.		
		updateOutput();
	}
	
	
	private void updateConstructInput() {
		cfg.controller.updateConstructInput();
	}

	private void updateOutput() {
		cfg.controller.updateOutput();
	}

	private void updateMergeResult() {
		cfg.controller.updateMergeResult();
	}

	private void updateReadMemory() {
		for (int i = 0; i < cfg.readHeads.length; i++) {
			DNCReadHead rh = cfg.readHeads[i];
			rh.updateReadMemory();
		}
	}

	private void updateReadHeadAddressing() {
		for (int i = 0; i < cfg.readHeads.length; i++) {
			DNCReadHead rh = cfg.readHeads[i];
			rh.updateReadHeadAddressing();
		}
	}

	private void updateWriteMemory() {
		cfg.writeHead.updateWriteMemory();
	}

	private void updateWriteMemoryLinkage() {
		cfg.memory.updateWriteMemoryLinkage();	
	}

	private void updateWriteHeadAddressing() {
		cfg.writeHead.updateWriteHeadAddressing();		
	}

	private void updateAllocateMemory() {
		cfg.memory.updateAllocateMemory();
	}

	private void updateGenerateInterfaceParameters() {
		cfg.writeHead.updateGenerateInterfaceParameters();
		for (int i = 0; i < cfg.readHeads.length; i++) {
			DNCReadHead rh = cfg.readHeads[i];
			rh.updateGenerateInterfaceParameters();
		}
	}

	private void updateProcessInput() {
		cfg.controller.updateProcessInput();	
	}
	
	public double getError() {
		return error;
	}
	
//	public double backWard(double [] output) {
//		backWard(output, null);
//	}

	double error;
	public double backWard(double [][] targets, int [] pos) {
		init = false;
		cfg.controller.reset4Bp();
		cfg.memory.reset4Bp();
		cfg.writeHead.reset4Bp(); 
		for (int i = 0; i < cfg.readHeads.length; i++) {
			cfg.readHeads[i].reset4Bp();
		}
		
		error = 0;		
		int cnt = 0;
		if (pos != null) {
			cnt = pos.length - 1;
		}
		
		double err = 0;
		lastT = t;
		for (int i = t; i >=0; i--) {
			t = i;
			double [] dyt = null;
			if (pos == null) {
				if (i == lastT) {
					//(6) regression or classify the results.
					dyt = backOutput(targets);
					error = error + cfg.controller.err;
				}	
			} else {
				if (cnt >= 0 && i == pos[cnt]) {
					dyt = backOutput(targets);
					cnt --;
					if (i == lastT) {
						error = error + cfg.controller.err;
					}					
				}
			}
					
			//(5) merge result from controller and read heads.
			backMergeResult(dyt);
			//(4) read from memory
			backReadMemory();
			//(3.4) read head addressing
			backReadHeadAddressing();
			//(3.3) write memory
			backWriteMemory();
			//(3.2) memory writing linkage
			backWriteMemoryLinkage();
			//(3.1) write memory addressing
			backWriteHeadAddressing();
			//(3) allocate memory usage
			backAllocateMemory();
			//(2) generate the interface params;
			double [] dv = backGenerateInterfaceParameters(cfg.controller.hts[t]);
			//(1.1) process input by controller 
			backProcessInput(new double [][] {{0}}, dv);
			//(1) construct input
			backConstructInput();
		}		
		
		
		return err;		
	}
	

	private void backConstructInput() {
		cfg.controller.backConstructInput();		
	}

	private void backProcessInput(double [][] x, double [] dr) {
		cfg.controller.backProcessInput(new double [][] {{0}}, dr);
	}

	private double [] backGenerateInterfaceParameters(double [] hts) {
		double [] dv = cfg.writeHead.backGenerateInterfaceParameters(hts);
		for (int i = 0; i < cfg.readHeads.length; i++) {
			DNCReadHead rh = cfg.readHeads[i];
			double [] dv1 = rh.backGenerateInterfaceParameters(hts);
			MathUtil.plus2V(dv1, dv);
		}
		
//		MathUtil.scale(dv, (double)1.0/(double)(1 + cfg.readHeads.length));
		DNCChecker.checkBg(dv, "backGenerateInterfaceParameters dv ", t);
//		if (MathUtil.isNaN(dv)) {
//			System.out.println("DV is not a number...");
//		}
		return dv;
	}




	private void backAllocateMemory() {
		cfg.memory.backAllocateMemory();		
	}




	private void backWriteHeadAddressing() {
		cfg.writeHead.backWriteHeadAddressing();
	}




	private void backWriteMemoryLinkage() {
		cfg.memory.backWriteMemoryLinkage();
	} 
	
	private void backWriteMemory() {
 		cfg.writeHead.backWriteMemory();
	} 

	private void backReadHeadAddressing() {
		for (int i = 0; i < this.cfg.readHeads.length; i++) {
			cfg.readHeads[i].backReadHeadAddressing();
		}
	}
 
	private void backReadMemory() {
		for (int i = 0; i < this.cfg.readHeads.length; i++) {
			cfg.readHeads[i].backReadMemory();
		}
	}
 
	public void backMergeResult(double [] dyt) {
		cfg.controller.backMergeResult(dyt);
	} 


	public double [] backOutput(double [][] output) { 
		return cfg.controller.backOutput(output);
	}
	
	

}
