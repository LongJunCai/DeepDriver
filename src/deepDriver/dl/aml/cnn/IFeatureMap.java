package deepDriver.dl.aml.cnn;

import java.io.Serializable;

import deepDriver.dl.aml.ann.IActivationFunction;

public interface IFeatureMap extends Serializable {
	
	public double[][] getFeatures();
	
	public double[][] getDeltaZzs();
	
	public boolean[][] getInitDeltaZzs();

	public void setInitDeltaZzs(boolean[][] initDeltaZzs);
	
	public double[][] getzZs();
	
	public IActivationFunction getAcf();
	
	public IConvolutionKernal[] getKernals();
	
	public double getbB();

	public void setbB(double bB);
	
	public double getDeltaBb();
	
	public void setDeltaBb(double deltaBb);

	public boolean isInitBb();

	public void setInitBb(boolean initBb);
	
	public void initData(IDataMatrix dm);
	
	public void resizeFeatures();
	
	public void reset();
	
	public double getU();

	public void setU(double u);

	public double getVar2();

	public void setVar2(double var2);

	public double getGema();

	public void setGema(double gema);

	public double getBeta();

	public void setBeta(double beta);
	
	public double getE();

	public void setE(double e);
	
	public double[][] getoZzs();

	public void setoZzs(double[][] oZzs);
	
	public double getDgamma();

	public void setDgamma(double dgamma);

	public double getDbeta();

	public void setDbeta(double dbeta);
	
	public double getSumU();

	public void setSumU(double sumU);

	public double getSumVar2();

	public void setSumVar2(double sumVar2);

	public int getSamplesCnt();

	public void setSamplesCnt(int samplesCnt);
	
	public Object[][] getLockObjs();

	public void setLockObjs(Object[][] lockObjs);
	
	public int[] getfMckIdMap();

	public void setfMckIdMap(int[] fMckIdMap);

}
