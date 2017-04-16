package deepDriver.dl.aml.dnc;

import java.util.List;

import deepDriver.dl.aml.ann.ANN;
import deepDriver.dl.aml.ann.INeuroUnit;
import deepDriver.dl.aml.ann.imp.NeuroUnitImp;
import deepDriver.dl.aml.lstm.BPTT;
import deepDriver.dl.aml.lstm.BPTT4MultThreads;
import deepDriver.dl.aml.lstm.BiBPTT;
import deepDriver.dl.aml.lstm.Context;
import deepDriver.dl.aml.lstm.ContextLayer;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.RNNNeuroVo;
import deepDriver.dl.aml.math.MathUtil;

public class DNCController {
	//lstm or bi-lstm, use headless lstm to implement this.
	LSTMConfigurator lstmCfg;
//	DNC dnc;
	BPTT bPTT;
	int htsLen;
	
	DNCConfigurator dcfg;
	DNCBPTT dbptt;
	
	double [][] wy;
	double [][] wr;
	
	double [][] dwy;
	double [][] dwr;
	
	double [][] dwy2;
	double [][] dwr2;
	
//	SoftMaxExp softMaxExp;
	ANN ann;
	
	double [] yt;
	double [][] hts;
	
	public DNCController(ANN ann, LSTMConfigurator cfg, DNCConfigurator dcfg) {
		super();
		this.ann = ann;
		this.lstmCfg = cfg;
		lstmCfg.setLearningRate(dcfg.getL());
		lstmCfg.setM(dcfg.getM());
		lstmCfg.setAutoSequence(false);
		lstmCfg.setBp4FirstLayer(true); 
		lstmCfg.setRequireLastRNNLayer(false);
		this.bPTT = createBPTT();
		bPTT.setGm(2);
		this.dcfg = dcfg;
		yt = new double[dcfg.yLen];
		htsLen = cfg.getLayers()[cfg.getLayers().length - 1].getRNNNeuroVos().length;
		wy = MathUtil.allocate(dcfg.yLen, htsLen);
		wr = MathUtil.allocate(dcfg.yLen, dcfg.rhNum * dcfg.memoryLength);
		dwy = MathUtil.allocate(dcfg.yLen, htsLen);
		dwr = MathUtil.allocate(dcfg.yLen, dcfg.rhNum * dcfg.memoryLength);
		dwy2 = MathUtil.allocate(dcfg.yLen, htsLen);
		dwr2 = MathUtil.allocate(dcfg.yLen, dcfg.rhNum * dcfg.memoryLength);
		MathUtil.initMatrix(wy, 1.0, -1.0);
		MathUtil.initMatrix(wr, 1.0, -1.0);
//		softMaxExp = new SoftMaxExp(outputLen, yt.length, 1.0);
		
		hts = new double[dcfg.maxTime][];
	} 
	
	public void prepareEnv() {
		MathUtil.reset2zero(yt);
		MathUtil.reset2zero(hts);
		reset4Bp();
	}

	public int getHtsLen() {
		return htsLen;
	}

	public void setHtsLen(int htsLen) {
		this.htsLen = htsLen;
	}
	
	public BPTT createBPTT() {
		if (lstmCfg.isBiDirection()) {
			return new BiBPTT(lstmCfg);
		} else {
			return new BPTT4MultThreads(lstmCfg);
		}
	}
	
	public void updateProcessInput() {
		bPTT.updateWws();
	}
	
	public void generateHts(ContextLayer [] cls, double [] x) {
		int t = dbptt.t;
//		ContextLayer cl = cls[cls.length - 1];
//		hts[t] = cl.getPreCxtAa();		
		
		int cnt = 0;
//		hts[t] = new double[cls[0].getPreCxtAa().length * cls.length + x.length];
		/***Use all the lstm layers
		hts[t] = new double[cls[0].getPreCxtAa().length * cls.length];
		for (int i = cls.length - 1; i >=0 ; i--) {			
			ContextLayer cl1 = cls[i];
			for (int j = 0; j < cl1.getPreCxtAa().length; j++) {
				hts[t][cnt ++] = cl1.getPreCxtAa()[j];
			} 
		}***/
		
		/****<START>Use the last layer only***/
		hts[t] = new double[cls[0].getPreCxtAa().length]; 
		ContextLayer cl1 = cls[cls.length - 1];
		for (int j = 0; j < cl1.getPreCxtAa().length; j++) {
			hts[t][cnt ++] = cl1.getPreCxtAa()[j];
		}  
		/****</END>Use the last layer only***/
//		for (int i = 0; i < x.length; i++) {
//			hts[t][cnt ++] = x[i];
//		}
		/**
		for (int i = 0; i < cl.getPreCxtAa().length; i++) {
			hts[t][cnt ++] = cl.getPreCxtAa()[i];
		}
		for (int i = 0; i < cl.getPreCxtSc().length; i++) {
			hts[t][cnt ++] = cl.getPreCxtSc()[i];
		}****/
	}
		
	public double [] processInput(double [][] x) {
		lstmCfg.setLearningRate(dcfg.getL());
		lstmCfg.setM(dcfg.getM());
		int t = dbptt.t;
		bPTT.setT(t);
		bPTT.settLength(t + 1);//since we do not konw the real kLength, so just assume it is.
		bPTT.fTT(x, false);
		Context cxt = bPTT.getHLContext();
		ContextLayer [] cls = cxt.getContextLayers();
		
		generateHts(cls, x[0]);
		
//		if (MathUtil.isNaN(hts[t]) && !b) {
//			System.out.println("LSTM produced NaN:"+ t);
//			print(hts[t]);
//			print(x[t]);
//			b = true;
//			System.out.println(MathUtil.isNaN(cls[0].getPreCxtAa()));
//			print(cls[0].getPreCxtAa());
//		}
//		if (MathUtil.isNaN(hts[t])) {
//			System.out.println("Reset the hts..");
//			MathUtil.reset2zero(hts[t]);
//			print(hts[t]);			
//		}
		return hts[t];
	}
	
	boolean b = true;
	
	public void print(double [] x) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < x.length; i++) {
			sb.append(x[i]+",");
		}
		System.out.println(sb.toString());
	}
	
	public void backProcessInput(double [][] x, double [] dr) { 
		int t = dbptt.t;
		bPTT.setT(t);
		bPTT.settLength(dbptt.lastT + 1);//as we know it should be this, so all should be changed. 
		Context deltaCxt = new Context();
		ContextLayer [] cxtLayers = new ContextLayer[this.bPTT.getLstmLayerLength()];
		for (int i = 0; i < cxtLayers.length; i++) {
			double [] dsc = new double[dr.length];
			double [] dAa = new double[dr.length];
			if (i == cxtLayers.length - 1) {
				for (int j = 0; j < dAa.length; j++) {
					dAa[j] = dr[j] + dhts[j];//all the dhts are sent here to process.
				}
			}
			cxtLayers[i] = new ContextLayer(dsc, dAa);
		}
		deltaCxt.setContextLayers(cxtLayers);
		bPTT.setDeltaCxt(deltaCxt);
		bPTT.bptt(x);
	}
	
	double [][] yts;
	int ytsCnt = 0;
	
	double[][] rss;
	
	public double[][] getRss() {
		return rss;
	}

	public void setRss(double[][] rss) {
		this.rss = rss;
	}
	
	public double [] output(int length, int [] pos) {
//		softMaxExp.compute(yt);
//		return softMaxExp.getR();
		int t = dbptt.t;
		if (pos == null) {
			return ann.forward(new double[][] {yt});
		} else {
			if (yts == null || pos[0] == t) {
				yts = MathUtil.allocate(pos.length, yt.length);
				ytsCnt = 0;
			} 
			for (int i = 0; i < yt.length; i++) {
				yts[ytsCnt][i] = yt[i];
			}
			ytsCnt++;
			if (t == length - 1) {
				ann.forward(yts);
				rss = ann.getRss();
				yts = null;
				ytsCnt = 0;				
			} 
			return null;
		}		
	}
	
	public void updateOutput() {
		ann.updateWws();
	}
	
	double [] dyt;
	double err;
	
	int botCnt = 0;
	public double [] backOutput(double [][] output) { 
		int t = dbptt.t;
		if (t == dbptt.lastT) {
			err = ann.bp(output, lstmCfg.getLearningRate(), lstmCfg.getM(), 0);
			botCnt = output.length - 1;
		}
		
		List<INeuroUnit> nuList = ann.getFirstLayer().getNeuros();
		
		if (dyt == null) {
			dyt = new double[yt.length];
		}
		for (int j = 0; j < nuList.size(); j++) {
			NeuroUnitImp nu = (NeuroUnitImp)nuList.get(j);
			dyt[j] = nu.getDeltaZ()[botCnt];
		}
		botCnt--;
		return dyt;
	}
	
	public void reset4Bp() {
		MathUtil.reset2zero(dhts);
		MathUtil.reset2zero(dyt);
		MathUtil.reset2zero(dwr);
		MathUtil.reset2zero(dwy);
	}
	
	public void backConstructInput() {
		
	}
	
	public void updateConstructInput() {
		
	}
	
	public double[] constructInput(double [] x) {
		int t = dbptt.t;
		double [] xt = new double[x.length + dcfg.readHeads.length * dcfg.memory.len];
		int cnt = 0;
		for (int i = 0; i < dcfg.readHeads.length; i++) {
			double [] rs_1 = new double[dcfg.readHeads[i].rs[0].length];
			if (t > 0) {
				rs_1 = dcfg.readHeads[i].rs[t - 1];
			}
			
			for (int j = 0; j < rs_1.length; j++) {
				xt[cnt ++] = rs_1[j];
			}
		}
		for (int i = 0; i < x.length; i++) {
			xt[cnt ++] = x[i];
		}
		return xt;
	}
	
	double [] dhts;
	
	public void backMergeResult2(double [] dyt) {
		int t = dbptt.t;
		
		if (dyt != null) {
			double [] rs = new double[dcfg.readHeads.length * dcfg.readHeads[0].rs[t].length];
			int cnt = 0;
			for (int i = 0; i < dcfg.readHeads.length; i++) {
				for (int j = 0; j < dcfg.readHeads[i].rs[t].length; j++) {
					rs[cnt ++] = dcfg.readHeads[i].rs[t][j];
				}
			}
			double [][] dwr1 = MathUtil.difMultipleX(dyt, rs); 
			double [][] dwy1 = MathUtil.difMultipleX(dyt, hts[t]); 
			
			if (dhts == null) {
				dhts = new double[hts[t].length];
			}
			
			MathUtil.plus(dwr, dwr1, dwr);
			MathUtil.plus(dwy, dwy1, dwy);
			cnt = 0;
			//need bp for hts in the 2nd step.  
			double [] drs = MathUtil.matrix2Vector(MathUtil.difMultipleY(dyt, wr)); 
			for (int i = 0; i < dcfg.readHeads.length; i++) {
				double [] drst = dcfg.readHeads[i].drs[t];
				for (int j = 0; j < drst.length; j++) {
					drst[j] = drs[cnt ++];
				}				
			}	
			
			
			double [] dhts0 = MathUtil.matrix2Vector(MathUtil.difMultipleY(dyt, wy));
			
			for (int i = 0; i < dhts.length; i++) {
				dhts[i] = dhts0[i];
			}
		} else {
			MathUtil.reset2zero(dhts);
		}
		
		//get from Xt
		if (t < dbptt.lastT) {
			int cnt = 0;
			RNNNeuroVo [] vos = lstmCfg.getLayers()[0].getRNNNeuroVos();
			double [] dxt = new double[vos.length];
			for (int i = 0; i < vos.length; i++) {
				dxt[i] = vos[i].getNvTT()[t + 1].getDeltaZz();
			}
			for (int i = 0; i < dcfg.readHeads.length; i++) {
				double [] drst = dcfg.readHeads[i].drs[t];
				for (int j = 0; j < drst.length; j++) {
					drst[j] = drst[j] + dxt[cnt ++];
				}				
			}
		}
	} 
	
	public void backMergeResult(double [] dyt) {
		int t = dbptt.t;
		
		if (dyt != null) {
			int cnt = 0;
			if (dhts == null) {
				dhts = new double[hts[t].length];
			}
			for (int i = 0; i < dcfg.readHeads.length; i++) {
				for (int j = 0; j < dcfg.readHeads[i].rs[t].length; j++) {
					dcfg.readHeads[i].drs[t][j] = dyt[cnt ++];
				}
			} 
			for (int i = 0; i < dhts.length; i++) {
				dhts[i] = dyt[cnt ++];
			}
		} else {
			MathUtil.reset2zero(dhts);
		}
		
		// get from Xt
		if (t < dbptt.lastT) {
			int cnt = 0;
			RNNNeuroVo[] vos = lstmCfg.getLayers()[0].getRNNNeuroVos();
			double[] dxt = new double[vos.length];
			for (int i = 0; i < vos.length; i++) {
				dxt[i] = vos[i].getNvTT()[t + 1].getDeltaZz();
			}
			for (int i = 0; i < dcfg.readHeads.length; i++) {
				double[] drst = dcfg.readHeads[i].drs[t];
				for (int j = 0; j < drst.length; j++) {
					drst[j] = drst[j] + dxt[cnt++];
				}
			}
		}
	}
	
	public void mergeResult() {
		int t = dbptt.t;
		double [] rs = new double[hts[t].length + dcfg.readHeads.length * dcfg.readHeads[0].rs[t].length];
		int cnt = 0;
		for (int i = 0; i < dcfg.readHeads.length; i++) {
			for (int j = 0; j < dcfg.readHeads[i].rs[t].length; j++) {
				rs[cnt ++] = dcfg.readHeads[i].rs[t][j];
			}
		} 
		for (int i = 0; i < hts[t].length; i++) {
			rs[cnt ++] = hts[t][i];
		}
		MathUtil.plus2V(rs, 1.0, yt, true);
	}
	
	
	public void mergeResult2() {
		int t = dbptt.t;
						
		double [] v1 = MathUtil.multipleV2v(wy, hts[t]);
		
		double [] rs = new double[dcfg.readHeads.length * dcfg.readHeads[0].rs[t].length];
		int cnt = 0;
		for (int i = 0; i < dcfg.readHeads.length; i++) {
			for (int j = 0; j < dcfg.readHeads[i].rs[t].length; j++) {
				rs[cnt ++] = dcfg.readHeads[i].rs[t][j];
			}
		}
		
		double [] v2 = MathUtil.multipleV2v(wr,  rs);		 
		MathUtil.plus2V(v1, v2, yt);
	}

	public void updateMergeResult() {
		//lots of delta...
		double l = lstmCfg.getLearningRate();
		double m = lstmCfg.getM(); 
		MathUtil.plus(dwr, -l, dwr2, m, dwr);		
		MathUtil.plus(wr, dwr, 1.0, wr);
		
		MathUtil.plus(dwy, -l, dwy2, m, dwy);		
		MathUtil.plus(wy, dwy, 1.0, wy);
//		MathUtil.plus(wy, dwy, -l, wy);
//		MathUtil.plus(wy, dwy2, m, wy);
		
		//deltaPara[i] = - l * deltaPara[i] + m * deltaPara2[i];
		MathUtil.set(dwr2, dwr);//It is should be done already.
		MathUtil.set(dwy2, dwy);
		
		MathUtil.reset2zero(dwr);
		MathUtil.reset2zero(dwy);
	}


}
