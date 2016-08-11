package deepDriver.dl.aml.lstm;

import java.io.Serializable;

public class Seq2SeqLSTMConfigurator implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	LSTMConfigurator qlSTMConfigurator;
	LSTMConfigurator alSTMConfigurator;
	int loop = 0;
	boolean testQ = true;
	public Seq2SeqLSTMConfigurator(LSTMConfigurator qlSTMConfigurator,
			LSTMConfigurator alSTMConfigurator) {
		super();
		this.qlSTMConfigurator = qlSTMConfigurator;
		this.alSTMConfigurator = alSTMConfigurator;
	}
	public LSTMConfigurator getQlSTMConfigurator() {
		return qlSTMConfigurator;
	}
	public void setQlSTMConfigurator(LSTMConfigurator qlSTMConfigurator) {
		this.qlSTMConfigurator = qlSTMConfigurator;
	}
	public LSTMConfigurator getAlSTMConfigurator() {
		return alSTMConfigurator;
	}
	public void setAlSTMConfigurator(LSTMConfigurator alSTMConfigurator) {
		this.alSTMConfigurator = alSTMConfigurator;
	}
	public boolean isTestQ() {
		return testQ;
	}
	public void setTestQ(boolean testQ) {
		this.testQ = testQ;
	}
	public int getLoop() {
		return loop;
	}
	public void setLoop(int loop) {
		this.loop = loop;
	}
	
	
	
}
