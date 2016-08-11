package deepDriver.dl.aml.lstm.distribution;


import deepDriver.dl.aml.lstm.NeuroNetworkArchitecture;
import deepDriver.dl.aml.lstm.Seq2SeqLSTM;
import deepDriver.dl.aml.string.ANFixedStreamImpV2;
import deepDriver.dl.aml.string.Dictionary;
import deepDriver.dl.aml.string.NFixedStreamImpV2;

public interface Seq2SeqLSTMBoostrapper {
	
	public void prepareData(boolean isServer) throws Exception;
	
	public void bootstrap(SimpleTask task, boolean need4Test) throws Exception;
	
	public NeuroNetworkArchitecture getNna();

	public void setNna(NeuroNetworkArchitecture nna);

	public NFixedStreamImpV2 getQsi();

	public void setQsi(NFixedStreamImpV2 qsi);

	public ANFixedStreamImpV2 getAsi();

	public void setAsi(ANFixedStreamImpV2 asi);

	public Seq2SeqLSTM getSeq2SeqLSTM();

	public void setSeq2SeqLSTM(Seq2SeqLSTM seq2SeqLSTM);

	public Dictionary getDic();
	
	public void setDic(Dictionary dic);
}
