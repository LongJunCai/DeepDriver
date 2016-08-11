package deepDriver.dl.aml.lstm;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class CxtLeverager implements IPreCxtProvider, ICxtConsumer, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	int providerIdx = 0;
	
	List<Context> arr = new ArrayList<Context>();
	int consumerIdx = 0;
	boolean complete = false;
	public void reset() {
		providerIdx = 0;
	}
	
	public boolean hasNext() {
		if(providerIdx <= arr.size() - 1) {
			return true;
		} 
		return false;
	}
	
	public Context next() {
		Context ctx = arr.get(providerIdx);
		providerIdx ++;
		return ctx;
	}

	@Override
	public void addContext(Context cxt) {
		if (!complete) {
			arr.add(cxt);
		} else {
			arr.set(consumerIdx, cxt);
		}
		
		if (requireObj != null) {
			synchronized (requireObj) {
				requireObj.notify();			
			}
			requireObj = null;
		}
		consumerIdx ++;
	}

	@Override
	public void complete() {
		System.out.println("There are "+consumerIdx+" contexts");
		complete = true;
		consumerIdx = 0;
	}

	@Override
	public boolean isCompleted() {
		return complete;
	}

	Object requireObj;
	@Override
	public void require(Object obj) {
		try {
			synchronized (obj) {
				obj.wait();
			}			
		} catch (InterruptedException e) {
			e.printStackTrace();
		}		
		requireObj = obj;
	}

}
