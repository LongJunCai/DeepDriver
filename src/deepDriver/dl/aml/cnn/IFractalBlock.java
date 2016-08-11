package deepDriver.dl.aml.cnn;

public interface IFractalBlock extends ICNNLayer {
	
	public int getCurrentDepth() ;
	
	public void setCurrentDepth(int currentDepth);
	
	public CNNLayer getDirectLayer() ;
	
	public void setDirectLayer(CNNLayer directLayer);
	
	public FractalBlock[] getFbs();
	
	public void setFbs(FractalBlock[] fbs);
	

}
