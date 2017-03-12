package deepDriver.dl.aml.dnc;

import deepDriver.dl.aml.math.MathUtil;

public class DNCMemory {
	
	DNCConfigurator cfg;
	DNCBPTT bptt;
	
	double [][][] memory;
	double [][][] dmemory;
	double [][] memory_1;
	
	int num;
	int len;
	
	double [][] ut; 
	double [][] dut; 
	
	double [][] fi; 
	double [][] dfi; 
	
	int [][] sut;
	double [][] a;
	double [][] da;
	double [] asum;
	
	double [][][] linkages;
	double [][][] dlinkages;
	double [][][] linkagesum;
	
	double [][] pt; 
	double [] ptsum; 
	double [][] dpt; 
	
	boolean [] bwLock;
	
	public DNCMemory(int num, int len, DNCConfigurator cfg) {
		this.num = num;
		this.len = len;
		memory = new double[cfg.getMaxTime()][][];
		dmemory = new double[cfg.getMaxTime()][][];
		memory_1 = MathUtil.allocate(num, len); 
		for (int i = 0; i < memory.length; i++) {
			memory[i] = MathUtil.allocate(num, len); 
			dmemory[i] = MathUtil.allocate(num, len); 
		}
		
		ut = MathUtil.allocate(cfg.getMaxTime(), num);
		dut = MathUtil.allocate(cfg.getMaxTime(), num);
		
		fi = MathUtil.allocate(cfg.getMaxTime(), num);
		dfi = MathUtil.allocate(cfg.getMaxTime(), num);
		sut = MathUtil.allocateInt(cfg.getMaxTime(), num);
		a = MathUtil.allocate(cfg.getMaxTime(), num);
		da = MathUtil.allocate(cfg.getMaxTime(), num);
		asum = new double[cfg.getMaxTime()];
		
		linkages = new double[cfg.getMaxTime()][][];
		dlinkages = new double[cfg.getMaxTime()][][];
		linkagesum = new double[cfg.getMaxTime()][][];
		for (int i = 0; i < linkages.length; i++) {
			linkages[i] = MathUtil.allocate(num, num); 
			dlinkages[i] = MathUtil.allocate(num, num); 
			linkagesum[i] = MathUtil.allocate(2, num); 
		}			
		
		bwLock = new boolean[cfg.getMaxTime()];
		
		pt = MathUtil.allocate(cfg.getMaxTime(), num);
		ptsum = new double[cfg.getMaxTime()];
		dpt = MathUtil.allocate(cfg.getMaxTime(), num);
		this.cfg = cfg;
	}
	
	public DNCBPTT getBptt() {
		return bptt;
	}

	public void setBptt(DNCBPTT bptt) {
		this.bptt = bptt;
	}
	
	

	public void reset4Bp() {
		MathUtil.reset2zero(dlinkages); 
		MathUtil.reset2zero(da); 
		MathUtil.reset2zero(dut); 
		MathUtil.reset2zero(dfi); 
		MathUtil.reset2zero(dmemory); 
		MathUtil.reset2zero(dpt);  
		MathUtil.reset(bwLock);
	}
	
	public void backAllocateMemory() {
		int t = bptt.t;
		
		for (int i = 0; i < sut[t].length; i++) {
			double t1 = 1.0;
			for (int j = 0; j < i; j++) {
				t1 = t1 * ut[t][sut[t][j]];
			}
//			a[t][sut[t][i]] = (1.0 - ut[t][sut[t][i]]) * t1;
			int l = sut[t][i];
			double t2 = (1.0 - ut[t][l]);
			dut[t][l] = dut[t][l] + da[t][l] * - t1;
			for (int j = 0; j < i; j++) { 
				int k = sut[t][j];
				if (ut[t][k] != 0) {
					dut[t][k] = dut[t][k] + da[t][l] * t2
						* t1/(ut[t][k]);
				} 				
			}
		}
		DNCChecker.checkBg(da[t], "Memory da ", t);
		boolean b0 = DNCChecker.checkBg(dut[t], "Memory dut1.1.", t);
		if (b0) {
			System.out.println("fuck...");
		}
		
		if (t < bptt.lastT) {
			double [] d2 = new double[cfg.writeHead.dwWs[t].length];
			double [] d3 = new double[cfg.writeHead.dwWs[t].length];
			MathUtil.multipleElements(dut[t + 1], fi[t + 1], d2);
			MathUtil.multipleElements(cfg.writeHead.wWs[t], d2, d3);
		
			MathUtil.plus2V(d2, 1.0, dut[t]);
			MathUtil.plus2V(d3, -1.0, dut[t]);
		}
		
		//dfi
		//initialize ut;
		double[] ut_1 = new double[ut[0].length];
		double [] utt = new double[ut[0].length];
		if (t > 0) {
			ut_1 = ut[t - 1];
		} 

		double[] wWs_1 = new double[cfg.writeHead.wWs[0].length];
		if (bptt.t > 0) {
			wWs_1 = cfg.writeHead.wWs[t - 1];
		}
		MathUtil.plus2V(ut_1, wWs_1, utt);
		double[] tm = new double[ut_1.length];
		MathUtil.multipleElements(ut_1, wWs_1, tm);
		MathUtil.plus2V(tm, -1.0, utt);
		MathUtil.multipleElements(utt, dut[t], dfi[t]);
		
		DNCChecker.checkBg(dut[t], "Memory dut", t);
		//wwt is caculated in the memory linkage already.
		if (t > 0) {
			for (int i = 0; i < cfg.readHeads.length; i++) {			
				DNCReadHead rh = cfg.readHeads[i]; 
				double [] v = new double[rh.wWs[0].length]; 
				MathUtil.plus2V(rh.wWs[t - 1], -rh.fgs[t], v);	
				MathUtil.plus(v, 1.0);			
				MathUtil.divideElements(fi[t], v, v);
				DNCChecker.checkBg(dfi[t], "Memory dfi", t);
				
				MathUtil.multipleElements(v, dfi[t], v);
				
				DNCChecker.checkBg(dfi[t], "Memory dfi", t);
				
				rh.dfgs[t] = - MathUtil.multiple(v, rh.wWs[t - 1]);
				DNCChecker.checkBg(rh.dfgs, "Memory rh.dfgs", t);
				//wrt is caculated during addressing phase.
			}
		}		
	}
	
	public void updateAllocateMemory() {
		
	}

	public void allocateMemory() {
		int t = bptt.t;
		//initialize the fi.
		for (int i = 0; i < cfg.readHeads.length; i++) {			
			DNCReadHead rh = cfg.readHeads[i]; 
			double [] v = new double[rh.wWs[0].length];
			if (t > 0) {
				MathUtil.plus2V(rh.wWs[t - 1], -rh.fgs[t], v);
			}
			
			MathUtil.plus(v, 1.0);			
			if (i == 0) {
				MathUtil.plus2V(v, 1.0, fi[t], true);
			} else {
				MathUtil.multipleElements(v, fi[t], fi[t]);
			}			
		}
		//initialize ut;
		double [] ut_1 = new double[ut[0].length];
		if (t > 0) {
			ut_1 = ut[t - 1];
		}
		MathUtil.plus2V(ut_1, 1.0, ut[t], true);
		
		double [] wWs_1 = new double[cfg.writeHead.wWs[0].length];
		if (t > 0) {
			wWs_1 = cfg.writeHead.wWs[t - 1];
		}
		MathUtil.plus2V(wWs_1, ut[t]);
		double [] tm = new double[ut_1.length];
		MathUtil.multipleElements(ut_1, wWs_1, tm);
		MathUtil.plus2V(tm, -1.0, ut[t]);
		
		MathUtil.multipleElements(fi[t], ut[t], ut[t]);
		
		DNCChecker.checkElementLess1(fi[t], "[fi] allocation", t);
		DNCChecker.checkElementLess1(ut[t], "[ut] allocation", t);
//		MathUtil.constrains(ut[t], 0, 1.0);
//		MathUtil.constrains(fi[t], 0, 1.0);
		
		//sort usage asc.
		int cnt = 0;
		boolean [] fgs = new boolean[ut[t].length];
		for (int i = 0; i < ut[t].length; i++) {
			int mi = 0;
			double min = ut[t][mi];
			for (int j = 0; j < ut[t].length; j++) {
				if (fgs[j]) {
					continue;
				}
				if (fgs[mi]) {
					mi = j;
					min = ut[t][j]; 
				} else { 
					if (min > ut[t][j]) {
						mi = j;
						min = ut[t][j]; 
					}					
				}
			}
			fgs[mi] = true;
			sut[t][cnt ++] = mi;
		}
		
		//initialize the allocation array
		for (int i = 0; i < sut[t].length; i++) {
			double t1 = 1.0;
			for (int j = 0; j < i; j++) {
				t1 = t1 * ut[t][sut[t][j]];
			}
			a[t][sut[t][i]] = (1.0 - ut[t][sut[t][i]]) * t1;
		}
		
		DNCChecker.checkSimplex(a[t], "[a] allocation", t);
		
//		DNCChecker.checkSimplex(a[t], "a "+t, t);		
		asum[t] = MathUtil.sumMaxK(a[t], MathUtil.K);
		MathUtil.simplex2(a[t], asum[t], MathUtil.K);
	}
	
	public void backWriteMemoryLinkage() { 
		int t = bptt.t;
		
		//update the delta for the memory linkage.
		if (t > 0) {
			for (int i = 0; i < cfg.readHeads.length; i++) {
				DNCReadHead rh = cfg.readHeads[i];
				MathUtil.plus(dlinkages[t], MathUtil.difMultipleX(rh.dfds[t], rh.wWs[t - 1]), 
						dlinkages[t]);
				MathUtil.plus(dlinkages[t], MathUtil.transpose(MathUtil.difMultipleX(rh.dbks[t], rh.wWs[t - 1])), 
						dlinkages[t]);
			}
		} 
		
		double [] wt = new double[cfg.writeHead.wWs[t].length];
		if (t < bptt.lastT) {
			wt = cfg.writeHead.wWs[t+1];
		} 
//		MathUtil.difMultipleY(dlinkages[t + 1], x)
		for (int i = 0; i < linkages[t].length; i++) {
			for (int j = 0; j < linkages[t][i].length; j++) {
//				if
				if (t < bptt.lastT) {
					dlinkages[t][i][j] = dlinkages[t][i][j] + (1.0 - wt[i] - wt[j])
						* dlinkages[t + 1][i][j];
					dpt[t][j] = dpt[t][j] + wt[i] * dlinkages[t + 1][i][j];	
				}
				
				if (t > 0) {
					cfg.writeHead.dwWs[t][i] = cfg.writeHead.dwWs[t][i]
							- linkages[t - 1][i][j] * dlinkages[t][i][j];
					cfg.writeHead.dwWs[t][j] = cfg.writeHead.dwWs[t][j]
							- linkages[t - 1][i][j] * dlinkages[t][i][j];

					cfg.writeHead.dwWs[t][i] = cfg.writeHead.dwWs[t][i]
							+ pt[t - 1][j] * dlinkages[t][i][j];// since pt0 =
																// 0;
				} 	
								
			}
		}
		
		/**
		 * Disable the linkage simplex for now.	*****/	
		MathUtil.difSimplex(linkages[t], linkagesum[t], dlinkages[t]);
		
		
		//update for the PT
		if (t < bptt.lastT) {
			double sum = MathUtil.sum(wt);
			MathUtil.plus2V(dpt[t + 1], (1.0 - sum), dpt[t], false); 
		}
		MathUtil.difSimplex(pt[t], ptsum[t], dpt[t]);
		
		//update for the wWs
		//gradient from Memory.
		double [] vk = MathUtil.matrix2Vector(MathUtil.difMultipleX(cfg.memory.dmemory[t], 
				new double [][] {cfg.writeHead.vts[t]}));
//		MathUtil.plus2V(vk, 1.0, cfg.writeHead.dwWs[t], true);
		MathUtil.plus2V(vk, 1.0, cfg.writeHead.dwWs[t]);
		
		DNCChecker.checkBg(cfg.memory.dmemory[t], "cfg.memory.dmemory[t] ", t);
		DNCChecker.checkBg(cfg.writeHead.vts[t], "Memory cfg.writeHead.vts[t] ", t);
		DNCChecker.checkBg(vk, "Memory vk ", t);
		
		if (t > 0) {
			double [][] d0 = MathUtil.allocate(cfg.memory.num, cfg.memory.len);
			
			double [] v2 = new double[cfg.writeHead.etss[t].length];
			MathUtil.plus2V(cfg.writeHead.etss[t], -1.0, v2);
			MathUtil.multipleByElements(cfg.memory.dmemory[t], cfg.memory.memory[t - 1], d0);
			double [] v3 = MathUtil.matrix2Vector(MathUtil.difMultipleX(d0, new double [][] {v2}));
			MathUtil.plus2V(v3, 1.0, cfg.writeHead.dwWs[t]);
		}
		//gradient from pt
		MathUtil.plus2V(dpt[t], cfg.writeHead.dwWs[t]);
		if (t > 0) {
			double d1 = -1.0 * MathUtil.multiple(dpt[t], pt[t - 1]);
			MathUtil.plus(cfg.writeHead.dwWs[t], d1);
		}
		
		//gradient from usage.
		if (t < bptt.lastT) {
			double [] d2 = new double[cfg.writeHead.dwWs[t].length];
			double [] d3 = new double[cfg.writeHead.dwWs[t].length];
			MathUtil.multipleElements(dut[t + 1], fi[t + 1], d2);
			MathUtil.multipleElements(ut[t], d2, d3);
		
			MathUtil.plus2V(d2, 1.0, cfg.writeHead.dwWs[t]);
			MathUtil.plus2V(d3, -1.0, cfg.writeHead.dwWs[t]);
		}
		
		/**Take the gradient explosion, control the value of it.**/
		MathUtil.difSimplex(cfg.writeHead.wWs[t], cfg.writeHead.wWsum[t], cfg.writeHead.dwWs[t]);
//		MathUtil.difSimplex(cfg.writeHead.wWs[t], cfg.writeHead.wWsum[t] < 1.0 ? 1.0 : cfg.writeHead.wWsum[t], cfg.writeHead.dwWs[t]);
//		MathUtil.gm(cfg.writeHead.dwWs[t], 1.0);
	}
	
	public void updateWriteMemoryLinkage() {
		
	}
	
	
	public void writeMemoryLinkage() {
		int t = bptt.t;
		double [] pt_1 = new double[pt[0].length];
		if (t > 0) {
			pt_1 = pt[t - 1];
		}
		double [] wt = cfg.writeHead.wWs[t];
		double sum = MathUtil.sum(wt);
		MathUtil.plus2V(pt_1, (1.0 - sum), pt[t], true);
		MathUtil.plus2V(wt, 1.0, pt[t], false);
		
			
		ptsum[t] = MathUtil.sumMaxK(pt[t], MathUtil.K);
		MathUtil.simplex2(pt[t], ptsum[t], MathUtil.K);
		
		boolean b = DNCChecker.checkSimplex(pt[t], "pt mlinkage", t);
		if (b) {
			System.out.println("What happened to pt");
		}
		/**
		double s = MathUtil.checkNormal(pt[t]);
		if (s < 0) {
			System.out.println("There is negative here in pts "+t);
		} else if (s < 1) {
		} else {
			System.out.println("The pts sum is: "+s);
		}***/
		
		
		for (int i = 0; i < linkages[t].length; i++) {
			for (int j = 0; j < linkages[t][i].length; j++) {
				if (t > 0) {
					linkages[t][i][j] = (1.0 - wt[i] - wt[j]) * linkages[t - 1][i][j] + wt[i] * pt_1[j];	
//					linkages[t][i][j] = linkages[t][i][j] ;
				} else {
					linkages[t][i][j] = wt[i] * pt_1[j];
				}
				
				if (i == j) {
					linkages[t][i][j] = 0;
				}
			}
		}
			
		/**
		 * Disable the linkage simplex for now.* ***/		
		MathUtil.simplex(linkages[t], linkagesum[t], MathUtil.K);
        
		
//		DNCChecker.checkSimplex(linkages[t], "linkage", t);
	}
	

}
