package deepDriver.dl.aml.lstm;

import java.io.Serializable;

public class LayerCfg implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	int attentionLength;

	public int getAttentionLength() {
		return attentionLength;
	}

	public void setAttentionLength(int attentionLength) {
		this.attentionLength = attentionLength;
	} 

}
