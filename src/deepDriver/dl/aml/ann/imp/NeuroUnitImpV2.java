package deepDriver.dl.aml.ann.imp;

public class NeuroUnitImpV2 extends NeuroUnitImp {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	LayerImp layer;
	public NeuroUnitImpV2(LayerImp layer) {
		super();
		this.layer = layer;
	}
	
	protected void initTheta() {
		double b = Math.pow(6.0/(double)(layer.getNeuros().size() + 
				layer.getPreviousLayer().getNeuros().size()), 0.5);
		length = 2*b;
		min = -b;
		max = b;
		if (randomize) {
			for (int i = 0; i < thetas.length; i++) {
				thetas[i] = length * random.nextDouble()
					+ min;
			}
		}		
	}
	

}
