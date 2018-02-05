package deepDriver.dl.aml.lstm;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;

public class BiProjectionLayer2 extends ProjectionLayer implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	static transient Random random = new Random(System.currentTimeMillis());
	int projectionLength = 150;
	int maxT = 0;
	int [] ws;
	Map<Integer, double []> w2vList = new HashMap<Integer, double []>();
	
	BiRNNNeuroVo [] vos1;		
	public BiProjectionLayer2(int projectionLength, int maxT, int nextLayerNN, LayerCfg lc) {
		super(projectionLength, maxT, nextLayerNN, lc);
		this.projectionLength = projectionLength;
		this.maxT = maxT;
		
		this.lc = lc;
		
		ws = new int[maxT];
		vos1 = new BiRNNNeuroVo[projectionLength];
		for (int i = 0; i < vos0.length; i++) {
			vos1[i] = new BiRNNNeuroVo(maxT, false, 0, 0, 0, nextLayerNN, lc);
		}
	}
	
	public void reverse(int lt) {
		for (int i = 0; i < vos1.length; i++) {
			vos1[i].reverse(lt);
		} 
	}
	
	public void reverseBack() {
		for (int i = 0; i < vos1.length; i++) {
			vos1[i].reverseBack();
		} 
	}
	
	LayerCfg lc;
	
	public LayerCfg getLc() {
		return lc;
	}

	public void setLc(LayerCfg lc) {
		this.lc = lc;
	}
	
	public void copyW2v2Pl(BiProjectionLayer2 pl) {
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
