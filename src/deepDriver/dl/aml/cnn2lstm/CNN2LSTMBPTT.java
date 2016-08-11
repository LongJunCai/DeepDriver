package deepDriver.dl.aml.cnn2lstm;


import deepDriver.dl.aml.cnn.CNNBP;
import deepDriver.dl.aml.cnn.CNNBP4MaxPooling;
import deepDriver.dl.aml.cnn.CNNConfigurator;
import deepDriver.dl.aml.cnn.ICNNLayer;
import deepDriver.dl.aml.cnn.IDataMatrix;
import deepDriver.dl.aml.cnn.IFeatureMap;
import deepDriver.dl.aml.lstm.BPTT;
import deepDriver.dl.aml.lstm.BPTT4MultThreads;
import deepDriver.dl.aml.lstm.IRNNLayer;
import deepDriver.dl.aml.lstm.IRNNNeuroVo;
import deepDriver.dl.aml.lstm.LSTMConfigurator;

public class CNN2LSTMBPTT {
	CNNConfigurator cnnCfg;
	LSTMConfigurator lstmCfg;
	
	CNNBP cNNBp;
	BPTT lstmBptt;	
	
	public CNN2LSTMBPTT(CNNConfigurator cnnCfg, LSTMConfigurator lstmCfg) {
		super();
		this.cnnCfg = cnnCfg;
		this.lstmCfg = lstmCfg;
		cNNBp = createCNNBP();
		lstmBptt = new BPTT4MultThreads(lstmCfg);
	}
	
	public CNNBP createCNNBP() {
		if (cnnCfg.getPoolingType() == CNNConfigurator.AVG_POOLING_TYPE) {
			return new CNNBP(cnnCfg);
		} else {
			return new CNNBP4MaxPooling(cnnCfg);
		}
	}
	
	public double runEpich(double [][] sample, 
			double [][] targets, IDataMatrix [] dataMatrix, double [] target) {
		double e = 0;
		cNNBp.fwd4(dataMatrix);
		ICNNLayer layer = cnnCfg.getLayers()[cnnCfg.getLayers().length - 1];
		double [] f2v = layer.featureMaps2Vector();
		double [][] nsample = new double[sample.length][];
		for (int i = 0; i < nsample.length; i++) {
			nsample[i] = new double[f2v.length + sample[i].length];
			for (int j = 0; j < f2v.length; j++) {
				nsample[i][j] = f2v[j];
			}
			for (int j = 0; j < sample.length; j++) {
				nsample[i][f2v.length + j] = sample[i][j];
			}
		}
		lstmBptt.fTT(nsample, false);
		
		lstmCfg.setBp4FirstLayer(true);
		e = lstmBptt.bptt(targets);
		//
		IRNNLayer rlayer = lstmCfg.getLayers()[0];
		IRNNNeuroVo [] vos = rlayer.getRNNNeuroVos();
		int cnt = 0;
		IFeatureMap [] fms = layer.getPreviousLayer().getFeatureMaps();
		for (int i = 0; i < fms.length; i++) {
			IFeatureMap fm = fms[i];
			double [][] deltaZzs = fm.getDeltaZzs();
			for (int j = 0; j < deltaZzs.length; j++) {
				for (int j2 = 0; j2 < deltaZzs[j].length; j2++) {
					IRNNNeuroVo vo = vos[cnt++];
					deltaZzs[j][j2] = vo.getNvTT()[0].getDeltaZz();
				}
			}
		}
		//
		cNNBp.bp();
		
		return e;		
	}

}
