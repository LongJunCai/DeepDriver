package deepDriver.dl.aml.cnn;

import java.io.Serializable;

public interface IDataStream extends Serializable {
	
	public IDataMatrix next();
	
	public IDataMatrix next(Object pos);
	
	public boolean hasNext();
	
	public boolean reset();
	
	public IDataStream [] splitStream(int segments);
	
	public int splitCnt(int segments);
}
