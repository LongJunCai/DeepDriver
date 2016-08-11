package deepDriver.dl.aml.lstm;

public class LSTMDataSet implements IStream {
	//sample--Time--features
	double [][][] samples;
	//sample--Time--targets
	double [][][] targets;
	
	public void reset() {
		cnt = -1;
	}
	
	public double[][][] getSamples() {
		return samples;
	}
	public void setSamples(double[][][] samples) {
		this.samples = samples;
	}
	public double[][][] getTargets() {
		return targets;
	}
	public void setTargets(double[][][] targets) {
		this.targets = targets;
	}
	
	int sampleTTLength;
	int sampleFeatureNum;
	int targetFeatureNum;
	
	public void init() {
//		if (condition) {
//			
//		}
		double [][] sample = samples[0];
		sampleTTLength = sample.length;
		double [] sampleFeature = sample[0];
		sampleFeatureNum = sampleFeature.length;
		double [] sampleTarget = targets[0][0];
		targetFeatureNum = sampleTarget.length;
	}
	int cnt = -1;
	@Override
	public boolean hasNext() {
		return cnt + 1 < samples.length;
	}
	@Override
	public void next() {
		cnt ++;
	}
	@Override
	public double[][] getSampleTT() {
		return samples[cnt];
	}
	@Override
	public double[][] getTarget() {
		return targets[cnt];
	}
	public int getSampleTTLength() {
		init();
		return sampleTTLength;
	}
	public int getSampleFeatureNum() {
		init();
		return sampleFeatureNum;
	}
	public int getTargetFeatureNum() {
		init();
		return targetFeatureNum;
	}

	@Override
	public Object getPos() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void next(Object pos) {
		// TODO Auto-generated method stub
		
	}

}
