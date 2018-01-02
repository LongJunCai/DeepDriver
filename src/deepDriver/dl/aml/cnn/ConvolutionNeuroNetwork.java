package deepDriver.dl.aml.cnn;

import java.io.File;
import java.io.Serializable;
import java.util.List;

import deepDriver.dl.aml.cnn.distribution.CNNMaster;
import deepDriver.dl.aml.cnn.test.SingleResult;
import deepDriver.dl.aml.distribution.Fs;
import deepDriver.dl.aml.lrate.BoldDriverLearningRateManager;

public class ConvolutionNeuroNetwork implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	CNNConfigurator cfg;
	
	public CNNConfigurator getCfg() {
		return cfg;
	}

	public void setCfg(CNNConfigurator cfg) {
		this.cfg = cfg;
	}

	public boolean isDebug() {
		return debug;
	}

	public void setDebug(boolean debug) {
		this.debug = debug;
	}

	public void construct(CNNArchitecture architecture, CNNConfigurator cfg) {
		List<LayerConfigurator>  cfgs = architecture.getLayerCfgs();
		this.cfg = cfg;		
		cfg.layers = new ICNNLayer[cfgs.size()];
		for (int i = 0; i < cfgs.size(); i++) {
			LayerConfigurator lc = cfgs.get(i);
			lc.setLast(i == cfgs.size() - 1);
			if (i == 0) {
				cfg.layers[i] = createCNNLayer(lc, null);
			} else {
				cfg.layers[i] = createCNNLayer(lc, cfg.layers[i - 1]);
			}			
		}
		
	}
	
	   public void enableTest(boolean enable) {
	        for (int i = 0; i < cfg.layers.length; i++) {
	            if (cfg.layers[i].getLc().getaNNCfg() != null) {
	                cfg.layers[i].getLc().getaNNCfg().setTesting(enable);
	            }           
	        }
	    }

	public ICNNLayer createCNNLayer(LayerConfigurator lc, ICNNLayer previous) {
		ICNNLayer layer = null;
		if (LayerConfigurator.CONVOLUTION_LAYER == lc.getType()) {
			layer = new CNNLayer(lc, previous);
		} else if (LayerConfigurator.POOLING_LAYER == lc.getType()) {
			layer = new SamplingLayer(lc, previous);
		} else if (LayerConfigurator.ANN_LAYER == lc.getType()) {
			layer = new CNNLayer2ANNAdapter(lc, previous);
		}
		return layer;
	}
	
	CNNBP cNNBP;
	double error = 0;
	boolean firstRnn = true;
	
	BoldDriverLearningRateManager lrm = new BoldDriverLearningRateManager();
	
	public void adjustMl(double error, int loop) {
		if (loop == 2) {
			cfg.setL(cfg.getL() * 0.1);
			return;
		}
		if (lrm != null) {
			double l = lrm.adjustML(error, cfg.getL());
			cfg.setL(l);
			System.out.println("Use M"+cfg.getM()+", L"+cfg.getL());
		}
	}
		
	public CNNBP createCNNBP() {
		if (cfg.poolingType == CNNConfigurator.AVG_POOLING_TYPE) {
			return new CNNBP(cfg);
		} else {
			return new CNNBP4MaxPooling(cfg);
		}
	}
	
	boolean debug = false;
	
	CNNMaster cm = new CNNMaster();
	public void train(IDataStream is, IDataStream tis) throws Exception {
//		lrm.setErrSize(3); 
//		lrm.setFlatThreshold(0.05);
//		lrm.setDecreaseRate(0.1);		
		/*
		 * 
		 * **/
		if (cm.isSetup()) {
			cm.trainModel(is, tis, this);
			System.out.println("CNN is running in the Distribution env.");
			return;
		}
		/*
		 * 
		 * **/
		if (cNNBP == null) {
			cNNBP = createCNNBP();
		}
		int loop = 0;
		while (firstRnn || error > cfg.acc) {
			firstRnn = false;
			error = 0;
			is.reset();
			int cnt = 0;
			int correctCnt = 0;
			while (is.hasNext()) {				
				IDataMatrix [] dm = is.next();
				if (dm == null) {
                    continue;
                }
				cNNBP.runTrainEpich(dm, dm[MatrixTargetIndex].getTarget());//this is strongly assumption.
				error = error + cNNBP.getStdError();
				double [] results = cNNBP.getResult();
				if (check(results, dm[MatrixTargetIndex].getTarget())) {
					correctCnt ++;				
				}
				cnt ++;
				if (cnt % 4000 == 0) {
					System.out.println("Complete "+cnt+" training...");
				}		
				if (debug) {
					if (cnt % 100 == 0) {
					System.out.println("Complete with "+cnt+" samples, accurracy is "+ 
						(double)correctCnt/(double)cnt
						+" with std error is "+ error /(double)cnt);
					}
				}				
				
			}
			loop++;
			error = error /(double)cnt;
//			adjustMl(error, loop);
			double acc = (double)correctCnt/(double)cnt;
//			if (acc > 0.80 && loop%5 == 0) {                        //有修改
				saveCfg2File(cfg.getName() + "-"+ loop, this.cfg);   //有修改
//			}
			System.out.println("Complete loop"+loop +" with "+cnt+" samples, accurracy is "+ 
					acc
					+" with std error is "+ error);
			
			test(tis);
		}	
		
	}
	
	long currentTimestamp = System.currentTimeMillis();
	String cfgFileName = "cnnCfg";
	
	public void saveCfg2File(String middleName, Object cfg) throws Exception {
		String sf = System.getProperty("user.dir");		
		File dir = new File(sf, "data");
		dir.mkdirs();		
		File f = null;
		if (middleName == null) {
			f = new File(dir, cfgFileName+"_"+currentTimestamp+".m");
		} else {
			f = new File(dir, cfgFileName+"_"+currentTimestamp+
					"_"+middleName+".m");
		}		
		Fs.writeObject2File(f.getAbsolutePath(), cfg);
		System.out.println("Save cfg to "+f.getAbsolutePath());
	}
	
	public void readCfg(String file) throws Exception {
		this.cfg = (CNNConfigurator) Fs.readObjFromFile(file);
	}
	
	public void readCfg(CNNConfigurator cfg) throws Exception {
		this.cfg = cfg;
	}
	
	
	public boolean check(double [] ta, double [] tb) {
		if (getMaxPos(ta) == getMaxPos(tb)) {
			return true;
		}
		return false;
	}
	
	public int getMaxPos(double [] ta) {
		int pos = 0;
		for (int i = 0; i < ta.length; i++) {
			if (ta[i] > ta[pos]) {
				pos = i;
			}
		}
		return pos;
	}
	
	public static void main(String[] args) {
		double [] ta = {0, 1, 0, 0, 0, 0};
		double [] tb = {0.1, 0.2, 0.3, 0.4, 0.5, 0};
		double [] tc = {0.1, 0.2, 0.01, 0.1, 0.03, 0};
		
		ConvolutionNeuroNetwork cnn = new ConvolutionNeuroNetwork();
		
		System.out.println(cnn.check(ta, tb));
		System.out.println(cnn.check(ta, tc));
		
	}
	
	public static int MatrixTargetIndex = 0; 
	 
	public void test(IDataStream tis) {
		if (tis == null) {
			return;
		}
		if (cNNBP == null) {
			cNNBP = createCNNBP();
		}
		enableTest(true);
		tis.reset();
		int allCnt = 0;
		int correctCnt = 0;
		boolean verify = false;
		while (tis.hasNext()) {				
			IDataMatrix [] dm = tis.next();
			if (dm == null) {
                continue;
            }
			double [] targets = cNNBP.test(dm);
			allCnt ++;
			
		//
			dm[MatrixTargetIndex].setResult(getMaxPos(targets));
		//	System.out.println((int)dm.getResult());
		//
			
			if (dm[MatrixTargetIndex].getTarget() != null && check(targets, dm[MatrixTargetIndex].getTarget())) {
			    verify = true;
				correctCnt ++;				
			}
			if (verify && allCnt % 1000 == 0) {
				System.out.println("Complete "+allCnt+" testings, accuracy: "+(double)correctCnt/(double)allCnt);
			}				
		}
		if (verify) {
            System.out.println("Complete "+allCnt+" testings, accuracy: "+(double)correctCnt/(double)allCnt);
        }		
		enableTest(false);
	}
	
	
	public CNNBP getcNNBP() {
		return cNNBP;
	}

	public void setcNNBP(CNNBP cNNBP) {
		this.cNNBP = cNNBP;
	}

	public SingleResult predict(IDataStream tis) 
	{
	    SingleResult singleResult = new SingleResult();
	    if (tis == null) 
	    {
            System.out.println("IDataStream is null");
	        return null;
        }
	    if (cNNBP == null) 
	    {
            cNNBP = createCNNBP();
        }
	    tis.reset();
	    IDataMatrix [] dm = tis.next();
	    double [] targets = cNNBP.test(dm);
	    int label = getMaxPos(targets);
	    double prob = targets[label];
	    singleResult.setLabel(label);
	    singleResult.setProb(prob);
	    return singleResult;
	    
	}

}
