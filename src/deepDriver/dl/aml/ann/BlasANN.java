package deepDriver.dl.aml.ann;

public class BlasANN {
	ArtifactNeuroNetwork ann;
	double [][] aAs = null;
	double [][][] wWs = null;
	double [][][] dwWs = null;
	public void trainModel(InputParameters parameters) {
		int ln = getLayerNum();
		if (aAs == null) {
			aAs = new double[ln][];
			wWs = new double[ln][][];
			dwWs = new double[ln][][];
		}
		
				
	}
	
	public int getLayerNum() {
		ILayer layer = ann.getFirstLayer();
		return 0;
	}

}
