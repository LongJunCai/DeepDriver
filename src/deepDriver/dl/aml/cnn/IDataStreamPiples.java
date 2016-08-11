package deepDriver.dl.aml.cnn;

public interface IDataStreamPiples {
	
	public IDataMatrix [] next();
	
	public boolean hasNext();
	
	public boolean reset();

}
