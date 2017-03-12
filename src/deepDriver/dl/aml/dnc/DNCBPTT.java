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

	public double [] forward(double [] x) {
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
		} else {
			t ++;
		}
		
		double [] result = null;
		//(1) construct input
		double [] xt = constructInput(x);
		//(1.1) process input by controller
		double [] hts = processInput(xt);//how to ensure the xt's length fits controller.
		//(2) generate the interface params;
		generateInterfaceParameters(hts);
		//(3) allocate memory usage
		allocateMemory();
		//(3.1) write memory addressing
		writeHeadAddressing();
		//(3.2) memory writing linkage
		writeMemoryLinkage();
		//(3.3) write memory
		writeMemory();
		//(3.4) read head addressing
		readHeadAddressing();
		//(4) read from memory
		readMemory();
		//(5) merge result from controller and read heads.
		mergeResult();
		//(6) regression or classify the results.		
		result = output();
			
		return result;
	}
	
	
	
	
	public double [] output() {
		return cfg.controller.output();
	}
	
	public void mergeResult() {
		cfg.controller.mergeResult();		
	}
	
	public void readMemory() {
		for (int i = 0; i < this.cfg.readHeads.length; i++) {
			cfg.readHeads[i].readMemory();
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
		
		double err = 0;
		lastT = t;
		for (int i = t; i >=0; i--) {
			t = i;
			double [] dyt = null;
			if (pos == null) {
				if (i == lastT) {
					//(6) regression or classify the results.
					dyt = backOutput(targets[0]);
					error = error + cfg.controller.err;
				}	
			} else {
				if (i == pos[cnt]) {
					dyt = backOutput(targets[cnt]);
					cnt ++;
					error = error + cfg.controller.err;
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


	public double [] backOutput(double [] output) { 
		return cfg.controller.backOutput(output);
	}
	
	

}
