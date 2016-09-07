package deepDriver.dl.aml.lstm.attentionEnDecoder;

import java.io.Serializable;

import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LstmAttention;

public class AttentionCfg implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	LSTMConfigurator qcfg;
	LSTMConfigurator acfg;
	LstmAttention attention;
	String name = "attention";
	
	public AttentionCfg(LSTMConfigurator qcfg, LSTMConfigurator acfg,
			LstmAttention attention) {
		super();
		this.qcfg = qcfg;
		this.acfg = acfg;
		this.attention = attention;
	}	
	
	
	public String getName() {
		return name;
	}



	public void setName(String name) {
		this.name = name;
	}



	public LSTMConfigurator getQcfg() {
		return qcfg;
	}
	public void setQcfg(LSTMConfigurator qcfg) {
		this.qcfg = qcfg;
	}
	public LSTMConfigurator getAcfg() {
		return acfg;
	}
	public void setAcfg(LSTMConfigurator acfg) {
		this.acfg = acfg;
	}
	public LstmAttention getAttention() {
		return attention;
	}
	public void setAttention(LstmAttention attention) {
		this.attention = attention;
	}
	
	

}
