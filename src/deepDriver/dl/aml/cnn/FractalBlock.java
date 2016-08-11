package deepDriver.dl.aml.cnn;


public class FractalBlock extends CNNLayer implements IFractalBlock {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public FractalBlock(LayerConfigurator lc, ICNNLayer previousLayer) {
		this(lc, previousLayer, lc.getFblockDepth());
	}
	public FractalBlock(LayerConfigurator lc, ICNNLayer previousLayer, int cDepth) {
		super(lc, previousLayer);
		this.currentDepth = cDepth;
		if (cDepth > 1) {
			directLayer = new CNNLayer(lc, previousLayer);
			fbs = new FractalBlock[lc.getFblockLayerNum()];
			for (int i = 0; i < fbs.length; i++) {
				if (i == 0) {
					fbs[i] = new FractalBlock(lc, previousLayer, cDepth - 1);
				} else {
					fbs[i] = new FractalBlock(lc, fbs[i - 1], cDepth - 1);
				}				
			}
		} 
	}
	int currentDepth;
	CNNLayer directLayer;
	FractalBlock [] fbs;
	
	public int getCurrentDepth() {
		return currentDepth;
	}
	public void setCurrentDepth(int currentDepth) {
		this.currentDepth = currentDepth;
	}
	public CNNLayer getDirectLayer() {
		return directLayer;
	}
	public void setDirectLayer(CNNLayer directLayer) {
		this.directLayer = directLayer;
	}
	public FractalBlock[] getFbs() {
		return fbs;
	}
	public void setFbs(FractalBlock[] fbs) {
		this.fbs = fbs;
	}

}
