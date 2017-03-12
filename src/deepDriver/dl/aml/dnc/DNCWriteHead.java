package deepDriver.dl.aml.dnc;

import deepDriver.dl.aml.math.ContentBasedWeighting;
import deepDriver.dl.aml.math.IMatrixExp;
import deepDriver.dl.aml.math.LinearMatrixExp;
import deepDriver.dl.aml.math.MathUtil;
import deepDriver.dl.aml.math.OnePlusExp;
import deepDriver.dl.aml.math.SigmodExp;

public class DNCWriteHead {
	
//	private MatrixExp kw; 
	IMatrixExp kw;
	double [][] kws;
	double [][] dkws;
	
	private OnePlusExp beta;
	double [] betas;
	double [] dbetas;
	
	private SigmodExp [] ets;
	private IMatrixExp vt;
	double [][] etss;
	double [][] vts;
	
	double [][] detss;
	double [][] dvts;
	
	private SigmodExp ga;
	private SigmodExp gw;
	double [] gas;
	double [] gws;
	
	double [] dgas;
	double [] dgws;
	
	double [][] wWs; 
	double [][] dwWs; 
	double [] wWsum;
	
	double [][] cs;
	double [][] dc;
	
	DNCConfigurator cfg;	
	DNCBPTT bptt;
	
	public DNCWriteHead(DNCConfigurator cfg) {
		super();
		this.cfg = cfg;
		int htsLen = cfg.controller.getHtsLen();
		int meLen = cfg.memory.len;
		int meNum = cfg.memory.num;
		this.kw = createIMatrixExp4Memory(meLen, htsLen); 
		kws = MathUtil.allocate(cfg.maxTime, meLen);
		dkws = MathUtil.allocate(cfg.maxTime, meLen);
		this.beta = new OnePlusExp(htsLen);
		betas = new double[cfg.maxTime];
		dbetas = new double[cfg.maxTime];
		
		this.ets = new SigmodExp[meLen];
		etss = MathUtil.allocate(cfg.maxTime, ets.length);
		detss = MathUtil.allocate(cfg.maxTime, ets.length);
		for (int i = 0; i < ets.length; i++) {
			ets[i] = new SigmodExp(htsLen);
		}				
		this.vt = createIMatrixExp4Memory(meLen, htsLen);
		vts = MathUtil.allocate(cfg.maxTime, meLen);
		dvts = MathUtil.allocate(cfg.maxTime, meLen);
		
		this.ga = new SigmodExp(htsLen);
		this.gw = new SigmodExp(htsLen);
		gas = new double[cfg.maxTime];
		gws = new double[cfg.maxTime];
		dgas = new double[cfg.maxTime];
		dgws = new double[cfg.maxTime];
		
		wWs = MathUtil.allocate(cfg.getMaxTime(), meNum);
		dwWs = MathUtil.allocate(cfg.getMaxTime(), meNum);
		wWsum = new double[cfg.maxTime]; 
		
		cs = MathUtil.allocate(cfg.maxTime, meNum);
		dc = MathUtil.allocate(cfg.maxTime, meNum);
	}
	
	public IMatrixExp createIMatrixExp4Memory(int lNum, int htsLen) {
//		return new SigmodMatrixExp(lNum, htsLen);
		return new LinearMatrixExp(lNum, htsLen);
	}
	
	public void updateGenerateInterfaceParameters() {
		kw.update(cfg.getL(), cfg.getM());
		beta.update(cfg.getL(), cfg.getM());
				
		for (int i = 0; i < ets.length; i++) {
			ets[i].update(cfg.getL(), cfg.getM());
		}
		
		vt.update(cfg.getL(), cfg.getM());
		
		ga.update(cfg.getL(), cfg.getM());
		
		gw.update(cfg.getL(), cfg.getM());
	}

	public void generateInterfaceParameters(double [] hts) {
		int t = bptt.t;
		
		kw.compute(hts);
		for (int i = 0; i < kw.getRs().length; i++) {
			kws[t][i] = kw.getRs()[i];
		}
		beta.compute(hts); 
		betas[t] = beta.getR();
				
		for (int i = 0; i < ets.length; i++) {
			ets[i].compute(hts);
			etss[t][i] = ets[i].getR();
		}
		
		vt.compute(hts);
//		vts[t] = vt.getRs();
		MathUtil.plus2V(vt.getRs(), 1.0, vts[t], true);
		
		ga.compute(hts);
		gas[t] = ga.getR();
		
		
		gw.compute(hts);
		gws[t] = gw.getR();
		
//		if (gas[t] < 1.0E-5) {
//			System.out.println("The gas is too samlaller"+gas[t]+","+t);
//		}
	}
	 
	public void updateWriteHeadAddressing() {
		
	}
	
	ContentBasedWeighting cbw = new ContentBasedWeighting();
	public void addressing() {
		int t = bptt.t;
		
		if (t > 0) {
			cbw.setMatrix(cfg.memory.memory[t - 1]);
		} else {
			cbw.setMatrix(cfg.memory.memory_1);
		}
		cbw.weighting(this.kws[t], betas[t]);
		for (int i = 0; i < cbw.getSm().length; i++) {
			cs[t][i] = cbw.getSm()[i];
		}
		
		DNCChecker.checkSimplex(cs[t], "cs write "+t, t);

		MathUtil.plus2V(cfg.memory.a[t], this.gas[t], wWs[bptt.t], true);
		MathUtil.plus2V(cs[t], (1.0 - this.gas[t]), wWs[bptt.t]);
		/****Disable the gws for now, to test if it will work.****/
		MathUtil.scale(wWs[t], gws[t]);
		
		DNCChecker.checkSimplex(wWs[t], "write Wws "+t, t);
		
		wWsum[t] = MathUtil.sumMaxK(wWs[t], MathUtil.K);
		MathUtil.simplex2(wWs[t], wWsum[t], MathUtil.K);

//		double s = MathUtil.checkNormal(wWs[t]);
//		if (s < 0) {
//			System.out.println("There is negative here in Wws" + t);
//		} else if (s < 1) {
//		} else {
//			System.out.println("The Wws sum is: " + s);
//		}
		
	}
	
	public double [] backGenerateInterfaceParameters(double [] hts) {
		int t = bptt.t;
		
		gw.difCompute(dgws[t], hts);
		double [] dv = new double[gw.getDv().length];
		MathUtil.plus2V(gw.getDv(), dv);
		gw.resetDv();
		
		ga.difCompute(dgas[t], hts);
		MathUtil.plus2V(ga.getDv(), dv);
		ga.resetDv();
		
		vt.difCompute(dvts[t], hts);
		MathUtil.plus2V(vt.getDv(), dv);
		vt.resetDv();
		
		for (int i = 0; i < ets.length; i++) { 
			ets[i].difCompute(detss[t][i], hts);
			MathUtil.plus2V(ets[i].getDv(), dv);
			ets[i].resetDv();
		}
		
		beta.difCompute(dbetas[t], hts);
		MathUtil.plus2V(beta.getDv(), dv);
		beta.resetDv();
		
		kw.difCompute(dkws[t], hts);
		MathUtil.plus2V(kw.getDv(), dv);
		kw.resetDv();
		
		return dv;
	}
	
	public void updateWriteMemory() {
		
	}
	
	public void writeMemory() {
		int t = bptt.t;
		
		double [] rs = new double[etss[t].length];
		for (int i = 0; i < ets.length; i++) {
			rs[i] = etss[t][i];
		}
		
		double [][] mt = cfg.memory.memory[t];
		
		double [][] vs = MathUtil.multiple(MathUtil.transpose(new double [][] {wWs[t]}), 
				new double [][] {vts[t]});

		if (t > 0) {
			double [][] mt_1 = cfg.memory.memory[t-1];
			double [][] e = MathUtil.allocateE(mt.length, mt[0].length);
			double [][] es = MathUtil.multiple(MathUtil.transpose(new double [][] {wWs[t]}), new double [][] {rs});
			MathUtil.minus(e, es, e);
			MathUtil.multipleByElements(e, mt_1, mt); 
			MathUtil.plus(mt, vs, mt);
		} else {
			MathUtil.set(mt, vs);
		}
		
	}
	
	public void backWriteMemory() {
		int t = bptt.t;
//		detss[t] = cfg.memory.dmemory[t]
		double [][] vk = MathUtil.difMultipleY(cfg.memory.dmemory[t], 
				MathUtil.transpose(new double [][] {wWs[t]}));
		
		MathUtil.plus2V(vk[0], 1.0, dvts[t], true);
		
		if (t > 0) {
			double [][] d0 = MathUtil.allocate(cfg.memory.num, cfg.memory.len);
			
			double [] v2 = new double[wWs[t].length];
			MathUtil.plus2V(wWs[t], -1.0, v2);
			MathUtil.multipleByElements(cfg.memory.dmemory[t], cfg.memory.memory[t - 1], d0);
			double [][] vi = MathUtil.difMultipleY(d0, 
					MathUtil.transpose(new double [][] {v2}));
			
			MathUtil.plus2V(vi[0], 1.0, detss[t], true);			
		}
		
	}
	
	public void reset4Bp() {
		MathUtil.reset2zero(dwWs); 
		
	}
	
	public void backWriteHeadAddressing() {
		int t = bptt.t;
		
		double [] d1 = new double[cfg.memory.a[t].length];		
		MathUtil.plus2V(cfg.memory.a[t], this.gas[t], d1, true);
		MathUtil.plus2V(cs[t], (1.0 - this.gas[t]),  d1);	 
		
		/****Disable the gws for now, to test if it will work.*****/
		dgws[t] = MathUtil.multiple(dwWs[t], d1);
//		dgws[t] = 0;
		
		double [] d2 = new double[cfg.memory.a[t].length];
		MathUtil.plus2V(dwWs[t], gws[t], d2);
		dgas[t] = MathUtil.multiple(d2, cfg.memory.a[t]);
		dgas[t] = dgas[t] - MathUtil.multiple(d2, cs[t]);
		MathUtil.plus2V(d2, gas[t], cfg.memory.da[t]);
		
		MathUtil.difSimplex(cfg.memory.a[t], cfg.memory.asum[t], cfg.memory.da[t]);
		
		DNCChecker.checkBg(dwWs[t], "Memory write head adressing dwWs[t]", t);
		DNCChecker.checkBg(gas, "Memory write head adressing gas", t);
		DNCChecker.checkBg(cfg.memory.da[t], "Memory write head adressing cfg.memory.da[t]", t);
		
		
		MathUtil.plus2V(d2, (1 - gas[t]), dc[t], true); 
		if (t > 0) {
			cbw.setMatrix(cfg.memory.memory[t - 1]);
		} else {
			cbw.setMatrix(cfg.memory.memory_1);
		}
			
//		cbw.weighting();
		cbw.backWeighting(dc[t], this.kws[t], betas[t]);//it is using mt_1
		dbetas[t] = cbw.getDbeta();
		MathUtil.plus2V(cbw.getDk(), 1.0, dkws[t], true);
		
		
	}
}
