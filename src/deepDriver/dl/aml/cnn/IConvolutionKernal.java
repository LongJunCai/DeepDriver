package deepDriver.dl.aml.cnn;

import java.io.Serializable;

public interface IConvolutionKernal extends Serializable {
	
	public int getFmapOfPreviousLayer();
	
	public void setFmapOfPreviousLayer(int fmapOfPreviousLayer);

}
