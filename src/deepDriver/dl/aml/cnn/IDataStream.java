package deepDriver.dl.aml.cnn;

public interface IDataStream {
	
	public IDataMatrix next();
	
	public IDataMatrix next(Object pos);
	
	public boolean hasNext();
	
	public boolean reset();
}
