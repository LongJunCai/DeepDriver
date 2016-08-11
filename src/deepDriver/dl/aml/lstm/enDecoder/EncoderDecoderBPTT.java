package deepDriver.dl.aml.lstm.enDecoder;

import deepDriver.dl.aml.lstm.BPTT;
import deepDriver.dl.aml.lstm.BPTT4MultThreads;
import deepDriver.dl.aml.lstm.Context;
import deepDriver.dl.aml.lstm.ContextLayer;
import deepDriver.dl.aml.lstm.IBlock;
import deepDriver.dl.aml.lstm.ICell;
import deepDriver.dl.aml.lstm.IForgetGate;
import deepDriver.dl.aml.lstm.IInputGate;
import deepDriver.dl.aml.lstm.IOutputGate;
import deepDriver.dl.aml.lstm.IRNNNeuroVo;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMLayer;
import deepDriver.dl.aml.lstm.SimpleNeuroVo;

public class EncoderDecoderBPTT {
	LSTMConfigurator qcfg;
	LSTMConfigurator acfg;	
	
	BPTT qBptt;
	BPTT aBptt;
	
	public EncoderDecoderBPTT(LSTMConfigurator qcfg, LSTMConfigurator acfg) {
		super();
		this.qcfg = qcfg;
		this.acfg = acfg;
		qBptt = new BPTT4MultThreads(qcfg);
		qBptt.setUseAbsoluteSc(true);
		aBptt = new BPTT4MultThreads(acfg);
		aBptt.setUseAbsoluteSc(true);
	}
	
	public void fTTEncoder(double [][] qsample) {
		qBptt.fTT(qsample, false);
		Context cxt = qBptt.getHLContext();
		aBptt.setPreCxts(cxt); 
	}
	
	public double [][] fTTDecoder(double [][] asample) { 
		return aBptt.fTT(asample, false);
	}
	
	public double [][] fTT(double [][] qsample, double [][] asample) {
//		qBptt.fTT(qsample, false);
//		Context cxt = qBptt.getHLContext();
//		aBptt.setPreCxts(cxt);
		fTTEncoder(qsample);
		return fTTDecoder(asample);
	}


	public double runEpich(double [][] qsample, 
			double [][] qtargets, double [][] asample, 
			double [][] atargets) {
		double e = 0;
		
		fTT(qsample, asample);
		
		e = aBptt.bptt(atargets);
		//bpa2q
		bpCxt();
		//
		qBptt.bptt(qtargets);
		aBptt.updateWws();
		qBptt.updateWws();
		return e;
	}
	
	public void bpCxt() {
		int fpos = aBptt.getFirstLstmPos();
		int lstmLayerLength = aBptt.getLstmLayerLength();
		Context deltaCxt = new Context();
		ContextLayer [] cxtLayers = new ContextLayer[lstmLayerLength];
		for (int i = 0; i < cxtLayers.length; i++) {
			cxtLayers[i] = getHLContextDelta(fpos + i); 
		}
		deltaCxt.setContextLayers(cxtLayers);
		qBptt.setDeltaCxt(deltaCxt);
	}
	
	public SimpleNeuroVo getIRNNNeuroVo(IRNNNeuroVo nv, int t) {
		return nv.getNvTT()[t];
	}
	
	public ContextLayer getHLContextDelta(int pos) {
		LSTMLayer lstmL = (LSTMLayer) acfg.getLayers()[pos];
		IRNNNeuroVo [] nvs = lstmL.getRNNNeuroVos();
		double [] rsc = new double[nvs.length];
		double [] aas = new double[nvs.length];
		int t = 0;
		IBlock [] blocks = lstmL.getBlocks();
		for (int i = 0; i < aas.length; i++) {
			double s = 0;			
			for (int k = 0; k < blocks.length; k++) {
				IBlock lastTBlock = blocks[k]; 
				IOutputGate outGate = lastTBlock.getOutPutGate();
				IInputGate inGate = lastTBlock.getInputGate();
				IForgetGate fGate = lastTBlock.getForgetGate();
				
				SimpleNeuroVo fVo_t = getIRNNNeuroVo(fGate, t); 	
				SimpleNeuroVo iVo_t = getIRNNNeuroVo(inGate, t);
				SimpleNeuroVo oVo_t = getIRNNNeuroVo(outGate, t); 
				s = s + fVo_t.getDeltaZz() * fGate.getLwWs()[i]
					+ iVo_t.getDeltaZz() * inGate.getLwWs()[i]
					+ oVo_t.getDeltaZz() * outGate.getLwWs()[i];
				ICell [] cells = lastTBlock.getCells();
				for (int l = 0; l < cells.length; l++) {
					ICell lastTCell = cells[l];
					s = s + lastTCell.getDeltaC()[t] * lastTCell.getLwWs()[i];
				}
			}
			aas[i] = s;
			
		}
		for (int k = 0; k < blocks.length; k++) {			
			IBlock lastTBlock = blocks[k]; 
			IInputGate inGate = lastTBlock.getInputGate();
			IForgetGate fGate = lastTBlock.getForgetGate();
			
			SimpleNeuroVo fVo_t = getIRNNNeuroVo(fGate, t); 	
			SimpleNeuroVo iVo_t = getIRNNNeuroVo(inGate, t);
			ICell [] cells = lastTBlock.getCells();
			for (int l = 0; l < cells.length; l++) {
				double deltaSc = 0;
				ICell lastTCell = cells[l];				
				deltaSc = deltaSc + lastTCell.getDeltaSc()[t] * fVo_t.getaA()
						+ fVo_t.getDeltaZz() * fGate.getRwWs()[l] + iVo_t.getDeltaZz() *
						inGate.getRwWs()[l];
				rsc[k * cells.length + l] = deltaSc;//1 cell in 1 block, otherwise this should be re-build
			}			
		}
		return new ContextLayer(rsc, aas);
	}

}
