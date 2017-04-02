package deepDriver.dl.aml.dnc;

import deepDriver.dl.aml.math.ContentBasedWeighting;
import deepDriver.dl.aml.math.IMatrixExp;
import deepDriver.dl.aml.math.LinearMatrixExp;
import deepDriver.dl.aml.math.MathUtil;
import deepDriver.dl.aml.math.OnePlusExp;
import deepDriver.dl.aml.math.SigmodExp;
import deepDriver.dl.aml.math.SoftMaxExp;

public class DNCReadHead {
	
	private IMatrixExp kr;
	double [][] krs;
	double [][] dkrs;
	
	private OnePlusExp beta;
	private SigmodExp fg;
	
	double [] betas;
	double [] dbetas;
	
	double [] fgs;
	double [] dfgs;
	
	private SoftMaxExp pi;
	double [][] pis;
	double [][] dpis;
	
	double [][] wWs;
	double [][] wWs4s;
	double [][] rs;
	double [] wWsum;
	
	double [][] dwWs;
	double [][] drs;
	
	double [][] cs;	
	double [][] dc;
	double [] csum;	
	
	double [][] fds;
	double [][] bks;
	double [] fdsum;
	double [] bksum;
	
	double [][] dfds;
	double [][] dbks;
	
	DNCConfigurator cfg;	
	DNCBPTT bptt;
	
	public DNCReadHead(DNCConfigurator cfg) {
		super();
		this.cfg = cfg;
		int htsLen = cfg.controller.getHtsLen();
		this.kr = createIMatrixExp4Memory(cfg.memory.len, htsLen);
		beta = new OnePlusExp(htsLen);
		fg = new SigmodExp(htsLen);
		
		int piLen = 3;
		pi = new SoftMaxExp(piLen, htsLen, 1.0);
		
		krs = MathUtil.allocate(cfg.maxTime, cfg.memory.len);
		dkrs = MathUtil.allocate(cfg.maxTime, cfg.memory.len);
		betas = new double[cfg.maxTime];
		dbetas = new double[cfg.maxTime];
		
		fgs = new double[cfg.maxTime];
		dfgs = new double[cfg.maxTime];
		
		pis = MathUtil.allocate(cfg.maxTime, piLen);
		dpis = MathUtil.allocate(cfg.maxTime, piLen);
		
		wWs = MathUtil.allocate(cfg.getMaxTime(), cfg.memory.num);
		wWs4s = MathUtil.allocate(cfg.getMaxTime(), cfg.memory.num);
		rs = MathUtil.allocate(cfg.getMaxTime(), cfg.memory.len);
		wWsum = new double[cfg.maxTime]; 
		fdsum = new double[cfg.maxTime];
		bksum = new double[cfg.maxTime];
		
//		wWs_1 = new double[cfg.memory.num];
//		rs_1 = new double[cfg.memory.len];
		cs = MathUtil.allocate(cfg.getMaxTime(), cfg.memory.num);
		csum = new double[cfg.maxTime];
		
		dwWs = MathUtil.allocate(cfg.getMaxTime(), cfg.memory.num);
		drs = MathUtil.allocate(cfg.getMaxTime(), cfg.memory.len);
		
//		dwWs_1 = new double[cfg.memory.num];
//		drs_1 = new double[cfg.memory.len];
		dc = MathUtil.allocate(cfg.getMaxTime(), cfg.memory.num);
		
		fds = MathUtil.allocate(cfg.getMaxTime(), cfg.memory.num);
		bks = MathUtil.allocate(cfg.getMaxTime(), cfg.memory.num);
		dfds = MathUtil.allocate(cfg.getMaxTime(), cfg.memory.num);
		dbks = MathUtil.allocate(cfg.getMaxTime(), cfg.memory.num);
	}
	
	public IMatrixExp createIMatrixExp4Memory(int lNum, int htsLen) {
//		return new SigmodMatrixExp(lNum, htsLen);
		return new LinearMatrixExp(lNum, htsLen);
	}
	
	public void updateGenerateInterfaceParameters() {
		double l = cfg.getL();
		double m = cfg.getM();
		kr.update(l, m);
		beta.update(l, m); 
		fg.update(l, m); 
		pi.update(l, m);
	}
	
	public void prepareEnv() {
		MathUtil.reset2zero(wWs);
		MathUtil.reset2zero(rs);
	}

	public void generateInterfaceParameters(double [] hts) {
		int t = bptt.t;
		
		kr.compute(hts);
		for (int i = 0; i < kr.getRs().length; i++) {
			krs[t][i] = kr.getRs()[i];
		}
		beta.compute(hts);
		betas[t] = beta.getR();
		fg.compute(hts);
		fgs[t] = fg.getR();
		pi.compute(hts);
		for (int i = 0; i < pi.getR().length; i++) {
			pis[t][i] = pi.getR()[i];
		} 
	}
	
	public double [] backGenerateInterfaceParameters(double [] hts) {
		int t = bptt.t;
		 
		pi.difCompute(dpis[t], hts);
		double [] dv = new double[pi.getDv().length];
		MathUtil.plus2V(pi.getDv(), dv);
		pi.resetDv();
		
		checkBg(dv, "pi", t);
		
		fg.difCompute(dfgs[t], hts);
		MathUtil.plus2V(fg.getDv(), dv);
		fg.resetDv();
		
		checkBg(fg.getX(), "fgx", t);
		checkBg(fg.getPara(), "fgPara", t);		
		if (checkBg(fg.getDl(), "fgDL", t)) {
			System.out.println("X is:");
			print(fg.getX());
			System.out.println("Para is:");
			print(fg.getPara());
			System.out.println(dfgs[t]);
		}
		checkBg(fg.getDl2(), "fgDL2", t);		
		checkBg(dv, "fg", t);
		
		beta.difCompute(dbetas[t], hts);
		MathUtil.plus2V(beta.getDv(), dv);
		beta.resetDv();
		
		checkBg(dv, "beta", t);
		
		kr.difCompute(dkrs[t], hts);
		MathUtil.plus2V(kr.getDv(), dv);
		kr.resetDv();
		
		checkBg(dv, "kr", t);
		
		return dv;
	}
	
	boolean bg = true;  
	public boolean checkBg(double [] x, String s, int t) {
		if (!bg) {			
			if (checkNormal(x, "rh_bg"+ s, t)) {
				bg = true;
				return bg;
			} 
		}
		return false;
	}
	
	
	public void reset4Bp() {
		MathUtil.reset2zero(dfds);
		MathUtil.reset2zero(dbks);
		MathUtil.reset2zero(drs); 
		MathUtil.reset2zero(dwWs); 
	}
	
	public void readMemory() { 
		int t = bptt.t;
		MathUtil.plus2V(MathUtil.multipleV2v(MathUtil.transpose(cfg.memory.memory[t]), 
				wWs[t]), 1.0, rs[t], true);
	}
	
	public void backReadMemory() {
		int t = bptt.t;
		
		double [] dws = MathUtil.difMultipleY2v(drs[t], MathUtil.transpose(cfg.memory.memory[t]));		
		MathUtil.plus2V(dws, 1.0, dwWs[t], true);
		if (t < bptt.lastT ) { 
			double [] dws1 = MathUtil.difMultipleY2v(dfds[t + 1],  cfg.memory.linkages[t + 1]);
			double [] dws2 = MathUtil.difMultipleY2v(dbks[t + 1], MathUtil.transpose(cfg.memory.linkages[t + 1])); 
			
			double [] v0 = new double[wWs[0].length];
			double [] dws3 = new double[wWs[0].length];
			for (int i = 0; i < cfg.readHeads.length; i++) {			
				DNCReadHead rh = cfg.readHeads[i]; 
				if (rh == this) {
					continue;
				}
				double [] v = new double[rh.wWs[0].length];
				
				MathUtil.plus2V(rh.wWs[bptt.t], -rh.fgs[t + 1], v);
				
				MathUtil.plus(v, 1.0);	
				if (i == 0) {
					MathUtil.plus2V(v, 1.0, v0, true);
				} else {
					MathUtil.multipleElements(v, v0, v0);
				}
			}
			MathUtil.scale(v0, -fgs[t + 1]);
			MathUtil.difMultipleElements(dws3, v0, cfg.memory.dfi[t + 1]);
			
			MathUtil.plus2V(dws1, 1.0, dwWs[t]);
			MathUtil.plus2V(dws2, 1.0, dwWs[t]);
			MathUtil.plus2V(dws3, 1.0, dwWs[t]);
			
			difSoftMax4Wws();
			
		} 
	}
	
	public void getResults() {
		
	}
	
	public void backReadHeadAddressing() { 
		int t = bptt.t;
		
		boolean bwLock = cfg.memory.bwLock[t];
		if (!bwLock) {
			cfg.memory.bwLock[t] = true;
		}
		double [][] dm = MathUtil.transpose(MathUtil.difMultipleX(drs[t], wWs[t]));		
		if (t < bptt.lastT && !bwLock) {	
			double [] ets = cfg.writeHead.etss[t + 1];
			double [] ers = new double[ets.length];
			for (int i = 0; i < ets.length; i++) {
				ers[i] = ets[i];
			}
			double [][] mt = cfg.memory.memory[t];
			double [][] e = MathUtil.allocateE(mt.length, mt[0].length);
			double [][] es = MathUtil.multiple(MathUtil.transpose(new double [][] {
					cfg.writeHead.wWs[t + 1]}), new double [][] {ers});
			MathUtil.minus(e, es, e);
			
			double [][] wm = MathUtil.allocate(cfg.memory.num, cfg.memory.len);
			MathUtil.difMultipleByElements(cfg.memory.dmemory[t + 1], e, wm);			
			MathUtil.plus(cfg.memory.dmemory[t], wm, cfg.memory.dmemory[t]); 
		} 
		DNCChecker.checkBg(dm, "read head dm ", t);
		MathUtil.plus(cfg.memory.dmemory[t], dm, cfg.memory.dmemory[t]);
		
		if (t > 0) {
			dpis[t][0] = MathUtil.multiple(dwWs[t], fds[t]);
			dpis[t][1] = MathUtil.multiple(dwWs[t], cs[t]);
			dpis[t][2] = MathUtil.multiple(dwWs[t], bks[t]);
			
			//TODO do we need to refine the when t = 0;
			MathUtil.plus2V(dwWs[t], pis[t][0], dfds[t], true);		
			MathUtil.plus2V(dwWs[t], pis[t][1], dc[t], true);
			MathUtil.plus2V(dwWs[t], pis[t][2], dbks[t], true);
		} else {
			dpis[t][1] = MathUtil.multiple(dwWs[t], cs[t]);
			MathUtil.plus2V(dwWs[t], pis[t][1], dc[t], true);
		}		
		
		
		
		/****Disable fd, and bk sum simplex.
		MathUtil.difSimplex(fds[t], fdsum[t], dfds[t]);
		MathUtil.difSimplex(bks[t], bksum[t], dbks[t]);***/
		
		
		/**Apply simplex to CS, disable it. 
		MathUtil.difSimplex(cs[t], csum[t], dc[t]);*****/
		
		cbw.setMatrix(cfg.memory.memory[t]);
//		cbw.weighting();
		cbw.backWeighting(dc[t], krs[t], betas[t]);
		
		double [][] dm2 = cbw.getDm();
		MathUtil.plus(dm2, cfg.memory.dmemory[t], cfg.memory.dmemory[t]);
		
		boolean bb = DNCChecker.checkBg(dm2, "read head dm2 ", t);
		if (bb) {
			System.out.println("<<begin");
			DNCChecker.print(dc[t]);
			DNCChecker.print(krs[t]);
			System.out.println(betas[t]);
			DNCChecker.print(cfg.memory.memory[t]);
			System.out.println("end>>");
		}
		
		if (t < bptt.lastT && !bwLock) {
			//Time t is not caculated yet, but time t+1 is caculated already.
			double [][] dm3 = cfg.writeHead.cbw.getDm();//it is caculate in write addressing round.
			
			DNCChecker.checkBg(dm3, "read head dm3 ", t);
			
			MathUtil.plus(dm3, cfg.memory.dmemory[t], cfg.memory.dmemory[t]);
		}
		
		
		dbetas[t] = cbw.getDbeta();
		MathUtil.plus2V(cbw.getDk(), 1.0, dkrs[t], true);
//		MathUtil.dif
		
	}
	
	ContentBasedWeighting cbw = new ContentBasedWeighting();
	
	public void readHeadAddressing() { 
		int t = bptt.t;
		double [] ws = wWs[t];
		if (t > 0) {
			double [] ws_1 = wWs[t - 1];
			MathUtil.plus2V(MathUtil.multipleV2v(cfg.memory.linkages[t], 
					ws_1),  1.0, fds[t], true);	
			
			MathUtil.plus2V(MathUtil.multipleV2v(MathUtil.transpose(cfg.memory.linkages[t]), 
					ws_1),  1.0, bks[t], true);	 
		} else {
			MathUtil.reset2zero(fds[t]);
			MathUtil.reset2zero(bks[t]);
		}
		
		cbw.setMatrix(cfg.memory.memory[t]);
		cbw.weighting(krs[t], betas[t]);
		for (int i = 0; i < cbw.getSm().length; i++) {
			cs[t][i] = cbw.getSm()[i];
		}	
		
		/**Apply simplex to CS
		csum[t] = MathUtil.sumMaxK(cs[t], MathUtil.K);
		MathUtil.simplex2(cs[t], csum[t], MathUtil.K);		
		DNCChecker.checkSimplex(cs[t], "cs read"+t, t);*****/
		 
		
		/****Disable fd, and bk sum simplex. 	
		fdsum[t] = MathUtil.sumMaxK(fds[t], MathUtil.K);
		MathUtil.simplex2(fds[t], fdsum[t], MathUtil.K);
		
		bksum[t] = MathUtil.sumMaxK(bks[t], MathUtil.K);
		MathUtil.simplex2(bks[t], bksum[t], MathUtil.K);
		
		DNCChecker.checkSimplex(fds[t], "fds "+t, t);
		DNCChecker.checkSimplex(bks[t], "bks "+t, t);* ***/	
		

		if (t > 0) {
			MathUtil.plus2V(fds[t], pis[t][0], ws, true);		
			MathUtil.plus2V(cs[t], pis[t][1], ws);
			MathUtil.plus2V(bks[t], pis[t][2], ws);
		} else {
			MathUtil.plus2V(cs[t], pis[t][1], ws, true);
		}
		
		DNCChecker.checkSimplex(ws, "Read Wws "+t, t);
		
		/****Disable read ww, it can not process last ones.
		wWsum[t] = MathUtil.sumMaxK(ws, MathUtil.K);
		MathUtil.simplex2(ws, wWsum[t], MathUtil.K);***/
		
		softMax4Wws();
		
		/***
		if (!b) {			
			// checkNormal(krs[t], "krs", t);
			boolean a1 = checkNormal(fds[t], "fds", t);
			boolean a2 = checkNormal(bks[t], "bks", t);
			boolean a3 = checkNormal(cs[t], "rcs", t);
			if (a3) {
				System.out.println("betas for read: " + betas[t]);
				System.out.println("The hts is: ");
				print(beta.getX());
				System.out.println("KRS: ");
				print(krs[t]);
				System.out.println("memory: ");
				print(cfg.memory.memory[t]);
			}
			boolean a4 = checkNormal(pis[t], "pis", t);
			boolean a5 = checkNormal(wWs[t], "Wrs", t);
			boolean [] as = new boolean [] {a1, a2, a3, a4, a5};
			for (int i = 0; i < as.length; i++) {
				if (as[i]) {
					b = true;
					break;
				}				
			}
		}***/
	}
	
	boolean useNormalize = true;
	
	public void softMax4Wws() {
		int t = bptt.t;
		if (useNormalize) {
			/**try with normalization***/
			wWsum[t] = MathUtil.sum(wWs[t]);
			MathUtil.normalize(wWs[t]);			
		} else {
			MathUtil.plus2V(wWs[t], 1.0, wWs4s[t], true);
			wWs[t] = MathUtil.softMax(wWs[t], 1.0);
		}
		
	}
	
	public void difSoftMax4Wws() {
		int t = bptt.t; 
		if (useNormalize) {
			/****Disable read ww, it can not process last ones.
			MathUtil.difSimplex(wWs[t], wWsum[t], dwWs[t]);*****/
			MathUtil.scale(dwWs[t], 1.0/wWsum[t]);
		} else {
			dwWs[t] = MathUtil.difSoftMax4Weighting(dwWs[t], wWs4s[t], 1.0);
		}		
	}
	
	boolean b = false;
	
	public void print(double [][] x) {
		print(x[0]);
//		for (int i = 0; i < x.length; i++) {
//			print(x[i]);
//		}
	}
	
	public void print(double [] x) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < x.length; i++) {
			sb.append(x[i]+",");
		}
		System.out.println(sb.toString());
	}
	
	public boolean checkNormal(double [] x, String name, int t) {
		boolean b = MathUtil.isNaN(x);
		if (b) {
			System.out.println(name +" is NaN");
			print(x);
		}
		return b;
	}

	public void updateReadHeadAddressing() { 
		
	}

	public void updateReadMemory() {
		
	}

}
