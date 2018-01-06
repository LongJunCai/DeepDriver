package deepDriver.dl.aml.cnn;

import deepDriver.dl.aml.common.distribution.Linkable;

public class CacheAbleDataStream implements IDataStream, Linkable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	IDataMatrix[][] idms;
	int cnt = 0;
	
	int pos;	
	Linkable next;

	public CacheAbleDataStream(int capacity) {
		super();
		this.idms = new IDataMatrix[capacity][];
	}		
	
	public int getCnt() {
		return cnt;
	}
	
	public void setCnt(int cnt) {
		this.cnt = cnt;
	}

	public Linkable getNext() {
		return next;
	}

	public void setNext(Linkable next) {
		this.next = next;
	}

	public void add(IDataMatrix[] idm) {
		idms[cnt ++] = idm;
	}

	@Override
	public IDataMatrix[] next() {
		return idms[pos++];
	}

	@Override
	public IDataMatrix[] next(Object pos) {
		int ip = ((Integer)pos).intValue();
		return idms[ip];
	}

	@Override
	public boolean hasNext() {
		return pos < cnt;
	}

	@Override
	public boolean reset() {
		pos = 0;
		return true;
	}

	@Override
	public IDataStream[] splitStream(int segments) {
		return null;
	}

	@Override
	public int splitCnt(int segments) {
		return 0;
	}


}
