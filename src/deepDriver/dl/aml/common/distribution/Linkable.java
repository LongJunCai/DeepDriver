package deepDriver.dl.aml.common.distribution;

public interface Linkable {
	
//	public Linkable nextLink();
	public Linkable getNext();

	public void setNext(Linkable next);

}
