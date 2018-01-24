package deepDriver.dl.aml.cnn;

import java.io.Serializable;


import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.costFunction.ICostFunction;

public class CNNLayer implements ICNNLayer, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	IFeatureMap [] featureMaps;
	LayerConfigurator lc;
	int fmNumber;	
	ICNNLayer previousLayer;
	IActivationFunction acf;
	public CNNLayer(LayerConfigurator lc, ICNNLayer previousLayer) {
		super();
		this.previousLayer = previousLayer;
		this.lc = lc;
		fmNumber = lc.getFeatureMapNum();
		
		if (!lc.isFullConnection) {
			//it is not the same matrix as the paper
			fmNumber = lc.getFeatureMapAllocationMatrix().length;
		} 
		featureMaps = new IFeatureMap[fmNumber];
		for (int i = 0; i < featureMaps.length; i++) {
			fmIndex = i;
			featureMaps[i] = createIFeatureMap();
		}
	}
	
	float [][] preM = null;
	int [][][] preIds = null;
	float [][] ckM = null;
	int [][][] ckIds = null;
	float [][] outM = null;
	int [][][] outIds = null;
	
	float [][] dpreM = null;
	float [][] dckM = null;
	float [][] dckM4Tmp = null;
	float [][] doutM = null;	
		
	public float[][] getDckM4Tmp() {
		return dckM4Tmp;
	}

	public void setDckM4Tmp(float[][] dckM4Tmp) {
		this.dckM4Tmp = dckM4Tmp;
	}

	public int[][][] getCkIds() {
		return ckIds;
	}

	public void setCkIds(int[][][] ckIds) {
		this.ckIds = ckIds;
	}

	public int[][][] getPreIds() {
		return preIds;
	}

	public void setPreIds(int[][][] preIds) {
		this.preIds = preIds;
	}

	public int[][][] getOutIds() {
		return outIds;
	}

	public void setOutIds(int[][][] outIds) {
		this.outIds = outIds;
	}

	public float[][] getPreM() {
		return preM;
	}

	public void setPreM(float[][] preM) {
		this.preM = preM;
	}

	public float[][] getCkM() {
		return ckM;
	}

	public void setCkM(float[][] ckM) {
		this.ckM = ckM;
	}

	public float[][] getOutM() {
		return outM;
	}

	public void setOutM(float[][] outM) {
		this.outM = outM;
	}

	public float[][] getDpreM() {
		return dpreM;
	}

	public void setDpreM(float[][] dpreM) {
		this.dpreM = dpreM;
	}

	public float[][] getDckM() {
		return dckM;
	}

	public void setDckM(float[][] dckM) {
		this.dckM = dckM;
	}

	public float[][] getDoutM() {
		return doutM;
	}

	public void setDoutM(float[][] doutM) {
		this.doutM = doutM;
	}

	public LayerConfigurator getLc() {
		return lc;
	}

	public void setLc(LayerConfigurator lc) {
		this.lc = lc;
	}

	int fmIndex = 0;
	
	public IFeatureMap createIFeatureMap() {
		int r = lc.getCkRows();
		int c = lc.getCkColumns();
		if (lc.getCks() != null) {
			r = lc.getCks()[fmIndex][0];
			c = lc.getCks()[fmIndex][1];
		}
		return new FeatureMap(this,
				lc.getAcf() == null? ActivationFactory.getAf().getReLU():lc.getAcf(), previousLayer, 
				r, c, lc.isFullConnection,
				lc.isFullConnection ? null : lc.getFeatureMapAllocationMatrix()[fmIndex], fmIndex);
	}
	
	public ICNNLayer getPreviousLayer() {
		return previousLayer;
	}

	public void setPreviousLayer(ICNNLayer previousLayer) {
		this.previousLayer = previousLayer;
	}
	
	public IFeatureMap[] getFeatureMaps() {
		return featureMaps;
	}
	public void setFeatureMaps(IFeatureMap[] featureMaps) {
		this.featureMaps = featureMaps;
	}

	@Override
	public double[] featureMaps2Vector() {
		double [][] f = featureMaps[0].getFeatures();
		double[] v = new double[featureMaps.length * 
		                        f.length * f[0].length];
		int cnt = 0;
		for (int i = 0; i < featureMaps.length; i++) {
			f = featureMaps[i].getFeatures();
			for (int j = 0; j < f.length; j++) {
				for (int j2 = 0; j2 < f[j].length; j2++) {
					v[cnt ++ ] = f[j][j2];
				}
			}
		}
		return v;
	}

	@Override
	public void accept(ICNNLayerVisitor visitor) {
		visitor.visitCNNLayer(this);		
	}
	
	ICostFunction costFunction;

	@Override
	public ICostFunction getCostFunction() {
		return costFunction;
	}	

}
