package deepDriver.dl.aml.cnn.distribution;

import deepDriver.dl.aml.cnn.CacheAbleDataStream;
import deepDriver.dl.aml.cnn.IDataMatrix;
import deepDriver.dl.aml.cnn.IDataStream;
import deepDriver.dl.aml.common.distribution.CommonSlave;
import deepDriver.dl.aml.distribution.ResourceMaster;

public class DataStreamDistUtil {
	
	int cnt;
	int cap = 4096;
	
	public int getCap() {
		return cap;
	}

	public void setCap(int cap) {
		this.cap = cap;
	}

	public int getCnt() {
		return cnt;
	}

	public void setCnt(int cnt) {
		this.cnt = cnt;
	}

	public void distributeDs(IDataStream is, int num) throws Exception {
		ResourceMaster rm = ResourceMaster.getInstance();
		is.reset();
		CacheAbleDataStream [] iss = new CacheAbleDataStream[num];
		for (int i = 0; i < iss.length; i++) {
			iss[i] = new CacheAbleDataStream(cap);
		}
		int i = 0;
		while (is.hasNext()) {
			cnt ++;
			IDataMatrix [] idm = is.next();
			CacheAbleDataStream ids = iss[i++]; 
			ids.add(idm);
			if (i > iss.length - 1) {
				i = 0;
				if (iss[i].getCnt() >= cap) {
					rm.distributeCommand(CommonSlave.CTASKPIECE);
					rm.distributeObjects(iss);
					
					iss = new CacheAbleDataStream[num];
					for (int j = 0; j < iss.length; j++) {
						iss[j] = new CacheAbleDataStream(cap);
					}					
				}
			}
			
		} 
		if (iss[0].getCnt() < cap) {
			rm.distributeCommand(CommonSlave.CTASKPIECE);
			rm.distributeObjects(iss);
		}		
	}

}
