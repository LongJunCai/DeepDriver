package deepDriver.dl.aml.common.distribution;

import deepDriver.dl.aml.cnn.CacheAbleDataStream;
import deepDriver.dl.aml.cnn.IDataMatrix;
import deepDriver.dl.aml.cnn.IDataStream;

public class LinkableDataStream implements IDataStream {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	CacheAbleDataStream root;
	CacheAbleDataStream current;
	
	public LinkableDataStream(CacheAbleDataStream root) {
		super();
		this.root = root;
		this.current = root;
	}

	@Override
	public IDataMatrix[] next() {
		return current.next();
	}

	@Override
	public IDataMatrix[] next(Object pos) {
		return current.next(pos);
	}

	@Override
	public boolean hasNext() {		
		if (!current.hasNext()) {
			current = (CacheAbleDataStream) current.getNext();
			if (current == null) {
				return false;
			} else {
				return current.hasNext();
			}
		} else {
			return true;
		}		
	}

	@Override
	public boolean reset() {
		this.current = root;
		CacheAbleDataStream cds = root;
		cds.reset();
		while (cds.getNext() != null) {
			cds = (CacheAbleDataStream) cds.getNext();
			cds.reset();
		}
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
