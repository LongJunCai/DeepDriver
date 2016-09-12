package deepDriver.dl.aml.w2v;

import java.io.Serializable;

public class KeyCntPair implements Serializable { 
	private static final long serialVersionUID = 1L;
	String key;
	double value;
	public String getKey() {
		return key;
	}
	public void setKey(String key) {
		this.key = key;
	}
	public double getValue() {
		return value;
	}
	public void setValue(double value) {
		this.value = value;
	} 
	
}