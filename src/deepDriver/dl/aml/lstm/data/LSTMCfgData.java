package deepDriver.dl.aml.lstm.data;

import java.io.Serializable;

public class LSTMCfgData implements Serializable {
	
	double [][][] cfg = null;
	int type;
	int loop;	

	public int getType() {
		return type;
	}

	public void setType(int type) {
		this.type = type;
	}

	public int getLoop() {
		return loop;
	}

	public void setLoop(int loop) {
		this.loop = loop;
	}

	public double[][][] getCfg() {
		return cfg;
	}

	public void setCfg(double[][][] cfg) {
		this.cfg = cfg;
	}
	

}
