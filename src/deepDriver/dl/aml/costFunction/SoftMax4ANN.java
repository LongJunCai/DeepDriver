package deepDriver.dl.aml.costFunction;

import java.io.Serializable;
import java.util.List;


import deepDriver.dl.aml.ann.ILayer;
import deepDriver.dl.aml.ann.INeuroUnit;
import deepDriver.dl.aml.ann.imp.LayerImp;
import deepDriver.dl.aml.ann.imp.NeuroUnitImp;

public class SoftMax4ANN implements ICostFunction, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	LayerImp layer;
	
	int zZIndex = 0;
	
	public int getzZIndex() {
		return zZIndex;
	}

	public void setzZIndex(int zZIndex) {
		this.zZIndex = zZIndex;
	}

	public double [] result1Ofk() {
		List<INeuroUnit> neuros = layer.getNeuros();
		double [] yt = new double[neuros.size()];
		for (int i = 0; i < neuros.size(); i++) {
			NeuroUnitImp nu = (NeuroUnitImp) neuros.get(i); 
			yt[i] = nu.getAas()[zZIndex];			
		}
		return yt;
	}
	
	public double getMaxZz() {
		List<INeuroUnit> neuros = layer.getNeuros();
		double max = 0;
		for (int i = 0; i < neuros.size(); i++) {
			NeuroUnitImp nu = (NeuroUnitImp) neuros.get(i);
			if (max < nu.getZzs()[zZIndex]) {
				max = nu.getZzs()[zZIndex];
			}
		}
		return max;
	}

	@Override
	public double [] activate() {
		List<INeuroUnit> neuros = layer.getNeuros();
		double [] yt = new double[neuros.size()];
		double sum = 0;
		double max = getMaxZz();
			
		for (int i = 0; i < neuros.size(); i++) {
			NeuroUnitImp nu = (NeuroUnitImp) neuros.get(i);
			yt[i] = Math.exp(nu.getZzs()[zZIndex] - max);
			sum = sum + yt[i];
		}		
		if (!(sum > 0)) {
			System.out.println("Sum is not bigger than 0, sum:"+sum+", max: "+max);
		} else {
//			System.out.println("It is normal...");
		}
		for (int i = 0; i < neuros.size(); i++) {
			NeuroUnitImp nu = (NeuroUnitImp) neuros.get(i);
//			vo.aA = yt[i]/sum;
			if ((sum > 0)) {
				nu.getAas()[zZIndex] = yt[i]/sum;
			} else {
				nu.getAas()[zZIndex] = 0;
			}			
			yt[i] = nu.getAas()[zZIndex];
		}
		return yt;
	}
	
	public double caculateStdError() {
		if (target == null) {
			return 0;
		}
		List<INeuroUnit> neuros = layer.getNeuros();
		double stdError = 0;
		if (true) {
			for (int i = 0; i < target.length; i++) {
				if (target[i] == 1) {
//					SimpleNeuroVo vo = vos[i].getNvTT()[t];	
					NeuroUnitImp nu = (NeuroUnitImp) neuros.get(i);
					stdError = - Math.log(nu.getAas()[zZIndex]);
				}
			}			
		}
		return stdError;
	}

	double [] target;
	@Override
	public void caculateCostError() {
		List<INeuroUnit> neuros = layer.getNeuros();
		double [] yt = new double[neuros.size()];
		double sum = 0;
		int k = 0;
		
		double max = getMaxZz();
		for (int i = 0; i < neuros.size(); i++) {
//			SimpleNeuroVo vo = vos[i].getNvTT()[t];
			NeuroUnitImp nu = (NeuroUnitImp) neuros.get(i);
			if (target[i] == 1) {
				k = i;					
			}
			yt[i] = Math.exp(nu.getZzs()[zZIndex] - max);
			sum = sum + yt[i];
		}		
//		yt = yt / sum;
		for (int i = 0; i < neuros.size(); i++) {
			NeuroUnitImp nu = (NeuroUnitImp) neuros.get(i);
//			vo.aA = yt[i]/sum;
//			nu.getAas()[zZIndex] = yt[i]/sum;	
			if ((sum > 0)) {
				nu.getAas()[zZIndex] = yt[i]/sum;
			} else {
				nu.getAas()[zZIndex] = 0;
			}	
			if (k == i) {
//				vo.deltaZz = vo.aA - (1.0) ;//* f.deActivate(vo.zZ);	
				nu.getDeltaZ()[zZIndex] = nu.getAas()[zZIndex] - 1.0;
			} else {
//				vo.deltaZz = vo.aA ;//* f.deActivate(vo.zZ);	
				nu.getDeltaZ()[zZIndex] = nu.getAas()[zZIndex];
			}
		}
	
	}

	public LayerImp getLayer() {
		return layer;
	}

	public void setLayer(ILayer layer) {
		this.layer = (LayerImp) layer;
	}

	public double[] getTarget() {
		return target;
	}

	public void setTarget(double[] target) {
		this.target = target;
	}
	
	public static void main(String[] args) {
		System.out.println(Math.exp(-1000));
	}

	@Override
	public double verfiyResult(double[] targets, double[] results) {
		int cnt = 0;
		for (int i = 0; i < targets.length; i++) {
			if (results[i] == targets[i]) {
				cnt ++;
			}
		}
		return (double)cnt/(double)targets.length;
	}
	
	
}
