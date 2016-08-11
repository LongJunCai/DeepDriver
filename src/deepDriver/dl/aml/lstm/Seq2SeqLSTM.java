package deepDriver.dl.aml.lstm;

import java.util.ArrayList;
import java.util.List;

import deepDriver.dl.aml.lstm.beamSearch.BeamLayer;
import deepDriver.dl.aml.lstm.beamSearch.BeamNode;
import deepDriver.dl.aml.lstm.beamSearch.BeamSearch;

public class Seq2SeqLSTM {
	Seq2SeqLSTMConfigurator cfg;
	public Seq2SeqLSTM(LSTM qlstm, LSTM alstm) {
		super();
		this.qlstm = qlstm;
		this.alstm = alstm;
		cfg = new Seq2SeqLSTMConfigurator
				(qlstm.cfg, alstm.cfg);
	}
	
	//ONLY happen in training cycle.
	public void setTrainCfg(Seq2SeqLSTMConfigurator cfg) {
		this.cfg = cfg;
		qlstm.setCfg(cfg.getQlSTMConfigurator());
		alstm.setCfg(cfg.getAlSTMConfigurator());
		qlstm.cfg.setName("q_lstm");
		alstm.cfg.setName("a_lstm");
		swith2TrainContextLvger();
	}
	
	public void rebuildCfg(Seq2SeqLSTMConfigurator cfg) {
		this.cfg = cfg;
		qlstm.rebuild(cfg.getQlSTMConfigurator());
		alstm.rebuild(cfg.getAlSTMConfigurator());
		qlstm.cfg.setName("q_lstm");
		alstm.cfg.setName("a_lstm");
		swith2TestContextLvger();
	}
	
	public void setTrainCfg(LSTMConfigurator cfg, boolean isQ) { 
		if (isQ) {
			qlstm.setCfg(cfg);
			qlstm.cfg.setName("q_lstm");
		} else {
			alstm.setCfg(cfg);		
			alstm.cfg.setName("a_lstm");
		}				
		swith2TrainContextLvger();
	}
	
	public LSTM getQlstm() {
		return qlstm;
	}

	public void setQlstm(LSTM qlstm) {
		this.qlstm = qlstm;
	}

	public LSTM getAlstm() {
		return alstm;
	}

	public void setAlstm(LSTM alstm) {
		this.alstm = alstm;
	}

	public Seq2SeqLSTMConfigurator getCfg() {
		return cfg;
	}

	LSTM qlstm = null; //new LSTM();
	LSTM alstm = null; //new LSTM();
	CxtLeverager4S2sTraining trainCxtLvger;
	CxtLeverager testCxtLvger = new CxtLeverager();
	public void trainModel(final IStream qds, final IStream ads,final NeuroNetworkArchitecture nna) throws InterruptedException {
		trainModel(true, qds, ads, nna);
	}
	boolean finishedQM = false;
	
	public boolean isFinishedQM() {
		return finishedQM;
	}

	public void setFinishedQM(boolean finishedQM) {
		this.finishedQM = finishedQM;
	}

	public void trainModel(boolean testBoth, final IStream qds, final IStream ads,final NeuroNetworkArchitecture nna) throws InterruptedException {
		qlstm.cfg.setName("q_lstm");
		alstm.cfg.setName("a_lstm");
		swith2TrainContextLvger();
		final boolean alstDone = false;
		if (testBoth) {	
			if (!finishedQM) {
				qlstm.trainModel(qds);
			} else {
				qlstm.trainModel(qds, true);
			}
		} else if (this.cfg.isTestQ()) {
			qlstm.trainModelWithBatchSize(qds);
		}
		
		Thread tq = new Thread() {
			@Override
			public void run() {
				
			}
		};
		Thread ta = new Thread() {
			public void run() {
//				alstDone = true;
			}
		};
		if (alstm.cfg.nna.isUseProjectionLayer()) {//if a has, the q must have
			ProjectionLayer apl = (ProjectionLayer) alstm.cfg.layers[LSTMConfigurator.ProjectionLayerID];
			ProjectionLayer qpl = (ProjectionLayer) qlstm.cfg.layers[LSTMConfigurator.ProjectionLayerID];
			qpl.copyW2v2Pl(apl);
		}
		if (testBoth) {			
			alstm.trainModel(ads);
			qlstm.cfg.setForceComplete(true);
		} else if (!this.cfg.isTestQ()) {
			alstm.trainModelWithBatchSize(ads);
		}
		
		tq.start();
		Thread.sleep(100);		
		ta.start();
		tq.join();
		ta.join();
	}
	
	public void swith2TrainContextLvger() {
		if (trainCxtLvger == null) {
			trainCxtLvger = new CxtLeverager4S2sTraining(qlstm, alstm);
		}
		alstm.cfg.setPreCxtProvider(trainCxtLvger);
		qlstm.cfg.setCxtConsumer(trainCxtLvger);
	}
	
	public void swith2TestContextLvger() {
		alstm.cfg.setPreCxtProvider(testCxtLvger);
		qlstm.cfg.setCxtConsumer(testCxtLvger);
	}
	
	
	public List<double [][][]> beamSearch(LSTMDataSet qds, LSTMDataSet ads,int sentenceNum, 
			int maxLengthPerSentence, double [] eofChar) throws InterruptedException {
		swith2TestContextLvger();
		LSTMDataSet oads = new LSTMDataSet();
		oads.setSamples(ads.getSamples());
		List<double [][][]> seqs = new ArrayList<double[][][]>();
		ads.setSamples(oads.getSamples());
		qlstm.testModel(qds); 
		BeamSearch bs = alstm.beamSearch(sentenceNum, ads, maxLengthPerSentence, eofChar);
		BeamLayer bl = bs.getLayers().get(bs.getLayers().size() - 1);
		List<BeamNode> bns = bl.getBns();
		for (int j = 0; j < bns.size(); j++) {
			seqs.add(new double[][][]{bs.getBnById(j, bl)});
		}
		return seqs;
	}
	
	public List<double [][][]> testModel(LSTMDataSet qds, LSTMDataSet ads,int sentenceNum, 
			int maxLengthPerSentence, double [] eofChar) throws InterruptedException {
		swith2TestContextLvger();
		LSTMDataSet oads = new LSTMDataSet();
		oads.setSamples(ads.getSamples());
		List<double [][][]> seqs = new ArrayList<double[][][]>();
		for (int i = 0; i < sentenceNum; i++) {
			ads.setSamples(oads.getSamples());
			qlstm.testModel(qds); 
			double [][][] result = alstm.testModel(ads, maxLengthPerSentence, eofChar);
			double [][][] nr = emliteEOF(result, eofChar);
			if (nr == null || nr.length <= 0 || nr[0].length <= 0) {
				System.out.println("No more context left, so can not continue");
				break;
			}
			qds.setSamples(nr);
			seqs.add(result);
		}		
		return seqs;
	}
	
	public double [][][] emliteEOF(double [][][] result, double [] eof) {
		double [][][] result1 = new double[result.length][][];
		for (int i = 0; i < result1.length; i++) {
			int l = 0;
			for (int j = 0; j < result[i].length; j++) {
				if (compareChar(result[i][j], eof)) {
					l = j;
					break;
				}
			}
			result1[i] = new double[l][];
			for (int j = 0; j < result1[i].length; j++) {
				result1[i][j] = result[i][j];
			}
		}
		return result1;
	}
	
	public boolean compareChar(double target [], double s []) {
		for (int i = 0; i < s.length; i++) {
			if (target[i] != s[i]) {
				return false;
			}
		}
		return true;
	}
	
	public static void main(String[] args) {
		Seq2SeqLSTM seq2SeqLSTM = new Seq2SeqLSTM(null, null);
		double [][][] re = new double[1][][];
		re[0] = new double[4][];
		for (int i = 0; i < re[0].length; i++) {
			re[0][i] = new double[3];
			for (int j = 0; j < re[0][i].length; j++) {
				re[0][i][j] = i + j;
			}
		}
		seq2SeqLSTM.emliteEOF(re, new double[]{2, 3, 4});
	}
		
	
}
