package deepDriver.dl.aml.costFunction;

import java.io.Serializable;

import deepDriver.dl.aml.ann.ILayer;
import deepDriver.dl.aml.ann.imp.LayerImp;
import deepDriver.dl.aml.ann.imp.NeuroUnitImp;
import deepDriver.dl.aml.ann.imp.NeuroUnitImpV3;
import deepDriver.dl.aml.math.MathUtil;

public class MTLCostFunction implements ICostFunction, Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	Task [] tks;
	int zZIndex; 
	LayerImp layer;
	

	public Task[] getTks() {
		return tks;
	}

	public void setTks(Task[] tks) {
		this.tks = tks;
	}

	@Override
	public int getzZIndex() {
		return zZIndex;
	}

	@Override
	public void setzZIndex(int zZIndex) {
		this.zZIndex = zZIndex;
	}

	@Override
	public double[] activate() {
		int cnt = 0;
		int l = 0;
		
		for (int i = 0; i < tks.length; i++) {
			Task tk = tks[i]; 
			l = l + tk.getNeuroLen();
		}
		double [] as = new double[l];
		int acnt = 0;
		
		
		
		for (int i = 0; i < tks.length; i++) {
			Task tk = tks[i]; 
			if (tk.getNus() == null) {} 
			tk.setNus(new NeuroUnitImp[tk.getNeuroLen()]); 
			tk.setzZs(new double[tk.getNus().length]);
			for (int j = 0; j < tk.getNus().length; j++) {
				tk.getNus()[j] = (NeuroUnitImp) layer.getNeuros().get(cnt ++); 
				/****GRL****/
				if (tk.getGrlStatus() != 0) {
					NeuroUnitImpV3 v3 = (NeuroUnitImpV3) tk.getNus()[j];
					v3.setGrl(tk.getGrlStatus());					
				}
				/****GRL****/
 			}
			
			 			
			for (int j = 0; j < tk.getNus().length; j++) { 
				tk.getzZs()[j] = tk.getNus()[j].getZzs()[zZIndex];
 			}
			
			
			if (Task.CF_SOFTMAX == tk.getCostType()) { 
				double [] aAs = MathUtil.softMax(tk.getzZs(), 1.0);
				for (int j = 0; j < aAs.length; j++) {
					tk.getNus()[j].getAas()[zZIndex] = aAs[j];
					as[acnt ++] = aAs[j];
				}
//				return aAs;
			} else if (Task.CF_STD == tk.getCostType()) {
				//if it is STD, activation is applied already.
//				double [] a = {};
				as[acnt ++] = tk.getNus()[0].getAas()[zZIndex];
			} 
		}
		return as;
	}
	
	double [] target;

	@Override
	public double caculateStdError() {
		double stdError = 0;
		int cnt = 0;
		for (int i = 0; i < tks.length; i++) {
			Task tk = tks[i]; 
			double [] result = new double[tk.getNeuroLen()];
			for (int j = 0; j < result.length; j++) {
				result[j] = target[cnt ++];
			}
			
			/******/
			if (!tk.checkRule(result)) {
				continue;
			}
			/******/
			
			if (Task.CF_SOFTMAX == tk.getCostType()) { 
				int k = MathUtil.getMaxPos(result);
				NeuroUnitImp nu = (NeuroUnitImp) tk.getNus()[k];
//				stdError = stdError + - Math.log(nu.getAas()[zZIndex]);
				/****GRL****/
				if (tk.getGrlStatus() != 0) { 
					stdError = stdError + -Math.abs(tk.getGrlStatus()) * 
							-Math.log(nu.getAas()[zZIndex]);				
				} else {
					stdError = stdError + -Math.log(nu.getAas()[zZIndex]);
				}
				/****GRL****/
			} else if (Task.CF_STD == tk.getCostType()) {
				NeuroUnitImp nu = (NeuroUnitImp) tk.getNus()[0];
				double a = nu.getAas()[zZIndex];
				double t = result[0];
				stdError = stdError + (a - t) * (a - t);
			}
		}
		return stdError;
	}

	@Override
	public void caculateCostError() {
		int cnt = 0;
		for (int i = 0; i < tks.length; i++) {
			Task tk = tks[i]; 
			double [] result = new double[tk.getNeuroLen()];
			for (int j = 0; j < result.length; j++) {
				result[j] = target[cnt ++];
			}
			
			
			if (Task.CF_SOFTMAX == tk.getCostType()) { 
				int k = MathUtil.getMaxPos(result);
				for (int j = 0; j < tk.getNus().length; j++) {
					NeuroUnitImp nu = (NeuroUnitImp) tk.getNus()[j];
					/******/
					if (!tk.checkRule(result)) {
						nu.getDeltaZ()[zZIndex] = 0;
						continue;
					}
					/******/
					if (k == j) {
						nu.getDeltaZ()[zZIndex] = nu.getAas()[zZIndex] - 1.0;
					} else {
						nu.getDeltaZ()[zZIndex] = nu.getAas()[zZIndex];
					}
				}				
			} else if (Task.CF_STD == tk.getCostType()) {
				NeuroUnitImp nu = (NeuroUnitImp) tk.getNus()[0];
				double a = nu.getAas()[zZIndex];
				double t = result[0];
				/******/
				if (!tk.checkRule(result)) {
					nu.getDeltaZ()[zZIndex] = 0;
					continue;
				}
				/******/
				nu.getDeltaZ()[zZIndex] = (a - t) * MathUtil.difSigmod(nu.getZzs()[zZIndex]);
			}
		}
	}

	@Override
	public void setLayer(ILayer layer) {
		this.layer = (LayerImp) layer;
	}

	@Override
	public void setTarget(double[] target) {
		this.target = target;
	}

	@Override
	public double verfiyResult(double[] targets, double[] results) {
		return 0;
	}

}
