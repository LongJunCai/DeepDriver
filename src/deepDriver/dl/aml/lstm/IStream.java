package deepDriver.dl.aml.lstm;

public interface IStream {
	
	public void reset();
	
	public boolean hasNext();
	
	public void next();
	
	public double [][] getSampleTT(); 
	
	public double [][] getTarget(); 
	
	public int getSampleTTLength();
	
	public int getSampleFeatureNum();
	
	public int getTargetFeatureNum();
	
	public Object getPos();
	
	public void next(Object pos);

	public IStream[] splitStream(int cnt);

	public int splitCnt(int cnt);
	
}
