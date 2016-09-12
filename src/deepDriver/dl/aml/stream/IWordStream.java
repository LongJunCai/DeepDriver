package deepDriver.dl.aml.stream;

public interface IWordStream {
	
	public void reset();
	
	public boolean hasNext();
	
	public void next();
	
	public String [] getSampleTT(); 
	
	public String [] getTarget(); 
	
	public int getSampleTTLength();
	
	public int getSampleFeatureNum();
	
	public int getTargetFeatureNum();
	
	public Object getPos();
	
	public void next(Object pos);

}
