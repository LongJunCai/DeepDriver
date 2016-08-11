package deepDriver.dl.aml.lstm;

public interface PreCxtProvider {
	
	public void reset();
	
	public boolean hasNext();
	
	public double [] next();
	
	public boolean isCompleted();
	
	public void require(Object obj);

}
