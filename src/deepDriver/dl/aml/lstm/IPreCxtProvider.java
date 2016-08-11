package deepDriver.dl.aml.lstm;

public interface IPreCxtProvider {
	
	public void reset();
	
	public boolean hasNext();
	
	public Context next();
	
	public boolean isCompleted();
	
	public void require(Object obj);

}
