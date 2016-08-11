package deepDriver.dl.aml.lstm;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;

public class ProjectionLayer implements IRNNLayer, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	static transient Random random = new Random(System.currentTimeMillis());
	int projectionLength = 150;
	int maxT = 0;
	int [] ws;
	Map<Integer, double []> w2vList = new HashMap<Integer, double []>();
	
	RNNNeuroVo [] vos0;		
	public ProjectionLayer(int projectionLength, int maxT, int nextLayerNN) {
		super();
		this.projectionLength = projectionLength;
		this.maxT = maxT;
		
		ws = new int[maxT];
		vos0 = new RNNNeuroVo[projectionLength];
		for (int i = 0; i < vos0.length; i++) {
			vos0[i] = new RNNNeuroVo(maxT, false, 0, 0, 0, nextLayerNN);
		}
	}
	
	public void copyW2v2Pl(ProjectionLayer pl) {
		pl.w2vList.clear();
		Iterator<Integer> iter = w2vList.keySet().iterator();
		while (iter.hasNext()) {
			Integer id = (Integer) iter.next();
			double [] v = w2vList.get(id);
			double [] v1 = new double[v.length];
			for (int i = 0; i < v1.length; i++) {
				v1[i] = v[i];
			}
			pl.w2vList.put(id, v1);
		}
	}
	
	public double [] generateV() {
		double [] v = new double[projectionLength];
		for (int i = 0; i < v.length; i++) {
			v[i] = random.nextDouble();
		}
		return v;
	}
	
	public int getWd(int t) {
		return ws[t];
	}

	public void setWd(int t, int wId) {
		ws[t] = wId;
	}

	@Override
	public void fTT(IBPTT bptt) {
		bptt.fTT4RNNLayer(this);
	}
	@Override
	public void bpTT(IBPTT bptt) {
		bptt.bpTT4RNNLayer(this);
	}
	
	public void updateWw(IRNNLayerVisitor bptt) {
		bptt.updateWw4RNNLayer(this);
	}

	@Override
	public RNNNeuroVo[] getRNNNeuroVos() {
		return vos0;
	}

	@Override
	public void setRNNNeuroVos(RNNNeuroVo[] rnnvos) {

	}

}
