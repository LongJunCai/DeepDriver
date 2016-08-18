package deepDriver.dl.aml.lstm.lstm2Ann;

import java.util.List;

import deepDriver.dl.aml.ann.ANN;
import deepDriver.dl.aml.ann.INeuroUnit;
import deepDriver.dl.aml.ann.imp.NeuroUnitImp;
import deepDriver.dl.aml.lstm.BPTT;
import deepDriver.dl.aml.lstm.BPTT4MultThreads;
import deepDriver.dl.aml.lstm.Context;
import deepDriver.dl.aml.lstm.ContextLayer;
import deepDriver.dl.aml.lstm.LSTMConfigurator;

public class Lstm2AnnBPTT {
	LSTMConfigurator qcfg;
	ANN ann;
	
	BPTT qBptt;

	public Lstm2AnnBPTT(LSTMConfigurator qcfg, ANN ann) {
		super();
		this.qcfg = qcfg;
		this.ann = ann;
		qBptt = new BPTT4MultThreads(qcfg);
	}
 
	public double [] fTT(double [][] sample) {
		qBptt.fTT(sample, false);
		Context cxt = qBptt.getHLContext();
		ContextLayer [] cls = cxt.getContextLayers();
		ContextLayer cl = cls[cls.length - 1];
		return ann.forward(new double[][] {cl.getPreCxtAa()});
	}
	
	public double bp(double [][] sample, double [][] result) {
		double err = ann.bp(result, qcfg.getLearningRate(), qcfg.getM(), 0);
		List<INeuroUnit> nuList = ann.getFirstLayer().getNeuros();
		Context deltaCxt = new Context();
		ContextLayer [] cxtLayers = new ContextLayer[this.qBptt.getLstmLayerLength()];
		for (int i = 0; i < cxtLayers.length; i++) {
			double [] dsc = new double[nuList.size()];
			double [] dAa = new double[nuList.size()];
			if (i == cxtLayers.length - 1) {
				for (int j = 0; j < dAa.length; j++) {
					NeuroUnitImp nu = (NeuroUnitImp)nuList.get(j);
					dAa[j] = nu.getDeltaZ()[0];
				}
			}
			cxtLayers[i] = new ContextLayer(dsc, dAa);
		}
		deltaCxt.setContextLayers(cxtLayers);
		qBptt.setDeltaCxt(deltaCxt);
		qBptt.bptt(sample);
		return err;
	}
	
	public double runEpich(double [][] sample, double [][] result) {
		double e = 0;
		
		fTT(sample);
		e = bp(sample, result);
		
		ann.updateWws();
		qBptt.updateWws();
		return e;
	}
	
	
	

}
