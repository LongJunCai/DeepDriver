package deepDriver.dl.aml.lrate;

public interface LearningRateManager {
	
	public double adjustML(double err, double lrate);

}
