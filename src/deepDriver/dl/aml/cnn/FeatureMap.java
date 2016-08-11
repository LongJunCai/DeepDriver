package deepDriver.dl.aml.cnn;

import java.io.Serializable;
import java.util.Random;


import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.random.RandomFactory;

public class FeatureMap implements IFeatureMap, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	IConvolutionKernal [] kernals;
	ICNNLayer previousLayer;
	
	ICNNLayer currentLayer;
	
//	static transient Random random = new Random(System.currentTimeMillis());
	transient Random random = RandomFactory.getRandom();
	
	int ckRows;
	int ckColumns;
	
	boolean isFullConnection;
	int [] previouFeatureMapSeq;
	
	IActivationFunction acf;
	
	double [][] features;
	double [][] deltaZzs;
	double [][] zZs;
	double [][] oZzs;
	boolean [][] initDeltaZzs;
	
	double bB;
	double deltaBb;
	boolean initBb;
	
	int fmIndex;		
	
	public double getbB() {
		return bB;
	}
	
	public void setbB(double bB) {
		this.bB = bB;
	}
	

	public double getDeltaBb() {
		return deltaBb;
	}

	public void setDeltaBb(double deltaBb) {
		this.deltaBb = deltaBb;
	}

	public boolean isInitBb() {
		return initBb;
	}

	public void setInitBb(boolean initBb) {
		this.initBb = initBb;
	}

	public FeatureMap(ICNNLayer currentLayer, IActivationFunction acf, ICNNLayer previousLayer, int ckRows, int ckColumns,
			boolean isFullConnection, int[] previouFeatureMapSeq, int fmIndex) {
		super();
		this.currentLayer = currentLayer;
		this.fmIndex = fmIndex;
		this.acf = acf;
		this.previousLayer = previousLayer;
		this.ckRows = ckRows;
		this.ckColumns = ckColumns;
		this.isFullConnection = isFullConnection;
		this.previouFeatureMapSeq = previouFeatureMapSeq;
		if (previousLayer == null) {
//			features = new double[ckRows][ckColumns];
			initFeatures(ckRows, ckColumns);	
			return;
		}		
		double [][] featureOfPrevious = previousLayer.getFeatureMaps()[0].getFeatures();
		resizeFeatures();
		/**
		double b = Math.pow(6.0/(double)(featureOfPrevious.length *
				featureOfPrevious[0].length + features.length * features[0].length), 0.5);
		length = 2*b;
		min = -b;**/
		if (!currentLayer.getLc().isCKAdaptive() && !currentLayer.getLc().isFMAdaptive()) {
			initWwBase();
		}		
		initCks();
		
	}	
	
	public IActivationFunction getAcf() {
		return acf;
	}

	public void setAcf(IActivationFunction acf) {
		this.acf = acf;
	}
	
	//used for BN
	double gema;
	double beta;
	
	double dgamma = 0;
	double dbeta = 0;
	
	double u;
	double var2;
	double e = 0.000001;
	
	double sumU;
	double sumVar2;
	int samplesCnt;	

	public double getSumU() {
		return sumU;
	}

	public void setSumU(double sumU) {
		this.sumU = sumU;
	}

	public double getSumVar2() {
		return sumVar2;
	}

	public void setSumVar2(double sumVar2) {
		this.sumVar2 = sumVar2;
	}

	public int getSamplesCnt() {
		return samplesCnt;
	}

	public void setSamplesCnt(int samplesCnt) {
		this.samplesCnt = samplesCnt;
	}

	public double getE() {
		return e;
	}

	public void setE(double e) {
		this.e = e;
	}

	public double getU() {
		return u;
	}

	public void setU(double u) {
		this.u = u;
	}

	public double getVar2() {
		return var2;
	}

	public void setVar2(double var2) {
		this.var2 = var2;
	}

	public double getGema() {
		return gema;
	}

	public void setGema(double gema) {
		this.gema = gema;
	}

	public double getBeta() {
		return beta;
	}

	public void setBeta(double beta) {
		this.beta = beta;
	}	

	public Object[][] getLockObjs() {
		return lockObjs;
	}

	public void setLockObjs(Object[][] lockObjs) {
		this.lockObjs = lockObjs;
	}
	
	class LockObj implements Serializable {
		private static final long serialVersionUID = 1L;				
	}

	boolean initB = false;
	Object [][] lockObjs;
	public void initFeatures(int r, int c) {
		features = new double[r][c];
		deltaZzs = new double[r][c];
		lockObjs = new Object[r][c];
		initDeltaZzs = new boolean[r][c];
		zZs = new double[r][c];
		oZzs = new double[r][c];
		for (int i = 0; i < features.length; i++) {
			features[i] = new double[c];
			deltaZzs[i] = new double[c];
			lockObjs[i] = new Object[c];
			initDeltaZzs[i] = new boolean[c];
		}
		for (int i = 0; i < lockObjs.length; i++) {
			for (int j = 0; j < lockObjs[i].length; j++) {
				lockObjs[i][j] = new LockObj();
			}
		}
		if (!initB) {
            this.bB = 0;//length * random.nextDouble() + min;
            initB = true;
            
            this.gema = length * random.nextDouble() + min;
            this.beta = 0;
        }				
	}
	
	boolean useDmDirectly = true;
	
	public void initData(IDataMatrix dm) {
		if (features.length != dm.getMatrix().length) {
			int r = dm.getMatrix().length;
			int c = dm.getMatrix()[0].length;
			initFeatures(r, c);
		}
		if (useDmDirectly) {
			features = dm.getMatrix();
		} else {
			for (int j = 0; j < features.length; j++) {
				for (int j2 = 0; j2 < features[j].length; j2++) {
					features[j][j2] = dm.getMatrix()[j][j2];
				}
			}
		}		
	}
	
	public void initWwBase() {
//		double [][] featureOfPrevious = previousLayer.getFeatureMaps()[0].getFeatures();
////		double b = Math.pow(6.0/(double)(
////				featureOfPrevious.length * featureOfPrevious[0].length
////				+  features.length * features[0].length), 0.5);
		double b = //(ckRows - 0.5) *
				Math.pow(6.0/(double)(ckRows * ckColumns * (
				previousLayer.getFeatureMaps().length + currentLayer.getFeatureMaps().length)), 0.5);
		length = 2*b;
		min = -b;
		max = b;
	}

	public void resizeFeatures() {
		double [][] featureOfPrevious = previousLayer.getFeatureMaps()[0].getFeatures();
		int padding = 2 * previousLayer.getLc().getPadding();
		//asume step = 1, and no need padding.
		int r = padding + featureOfPrevious.length - ckRows + 1;
		int c = padding + featureOfPrevious[0].length - ckColumns + 1;
		if (r <= 0) {//sometimes the ck is bigger...
			r = 1;
		}
		if (c <= 0) {
			c = 1;
		}
		if (features == null || r != features.length) {
            initFeatures(r, c);	
        }		
	}
	
	public void reset() {
		for (int i = 0; i < features.length; i++) {
			for (int j = 0; j < features[0].length; j++) {
				features[i][j] = 0;
				deltaZzs[i][j] = 0;
				initDeltaZzs[i][j] = false;
			}			
		}
	}
	
	double min = -1.0;
	double max = 1.0;
	double length = max - min;
	
	int [] fMckIdMap = null;		
	
	public int[] getfMckIdMap() {
		return fMckIdMap;
	}

	public void setfMckIdMap(int[] fMckIdMap) {
		this.fMckIdMap = fMckIdMap;
	}

	public void initCks() {
		if (isFullConnection) {
			IFeatureMap [] fms = previousLayer.getFeatureMaps();
			kernals = new IConvolutionKernal[fms.length];
			fMckIdMap = new int[fms.length];
			for (int i = 0; i < kernals.length; i++) {
				kernals[i] = createIConvolutionKernal();
				kernals[i].setFmapOfPreviousLayer(i);
				fMckIdMap[i] = i;
			}
		} else {
			int allocatedNum = 0;
			for (int i = 0; i < previouFeatureMapSeq.length; i++) {
				if (LayerConfigurator.FM_ALLOCATED == previouFeatureMapSeq[i]) {
					allocatedNum ++;
				}
			}
			kernals = new IConvolutionKernal[allocatedNum];
			fMckIdMap = new int[previouFeatureMapSeq.length];
			int j = 0;
			for (int i = 0; i < previouFeatureMapSeq.length; i++) {
				fMckIdMap[i] = -1;
				if (LayerConfigurator.FM_ALLOCATED == previouFeatureMapSeq[i]) {
					kernals[j] = createIConvolutionKernal();
					kernals[j].setFmapOfPreviousLayer(i);
					fMckIdMap[i] = j;
					j ++;
				}
			}
		}
	}
	
	public IConvolutionKernal createIConvolutionKernal() {
		ConvolutionKernal ck = new ConvolutionKernal();
		ck.wWs = new double[ckRows][ckColumns];
		ck.detalwWs = new double[ckRows][ckColumns];
		ck.initDeltaZzs = new boolean[ckRows][ckColumns];
		for (int i = 0; i < ck.wWs.length; i++) {
			ck.wWs[i] = new double[ckColumns];
			ck.detalwWs[i] = new double[ckColumns];
			ck.initDeltaZzs[i] = new boolean[ckColumns];
			for (int j = 0; j <ck.wWs[i].length ; j++) {
				ck.wWs[i][j] = length * random.nextDouble()
				+ min;				
			}
		}
		ck.b = length * random.nextDouble()
				+ min;
		return ck;
	}	
		
	public ICNNLayer getPreviousLayer() {
		return previousLayer;
	}

	public void setPreviousLayer(ICNNLayer previousLayer) {
		this.previousLayer = previousLayer;
	}

	public IConvolutionKernal[] getKernals() {
		return kernals;
	}	

	public double[][] getDeltaZzs() {
		return deltaZzs;
	}

	public void setDeltaZzs(double[][] deltaZzs) {
		this.deltaZzs = deltaZzs;
	}

	public void setKernals(IConvolutionKernal[] kernals) {
		this.kernals = kernals;
	}

	public double[][] getFeatures() {
		return features;
	}
	public void setFeatures(double[][] features) {
		this.features = features;
	}

	public boolean[][] getInitDeltaZzs() {
		return initDeltaZzs;
	}

	public void setInitDeltaZzs(boolean[][] initDeltaZzs) {
		this.initDeltaZzs = initDeltaZzs;
	}

	public double[][] getzZs() {
		return zZs;
	}

	public void setzZs(double[][] zZs) {
		this.zZs = zZs;
	}

	public double[][] getoZzs() {
		return oZzs;
	}

	public void setoZzs(double[][] oZzs) {
		this.oZzs = oZzs;
	}

	public double getDgamma() {
		return dgamma;
	}

	public void setDgamma(double dgamma) {
		this.dgamma = dgamma;
	}

	public double getDbeta() {
		return dbeta;
	}

	public void setDbeta(double dbeta) {
		this.dbeta = dbeta;
	}		
	
}
