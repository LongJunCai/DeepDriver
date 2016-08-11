package deepDriver.dl.aml.distribution;


public class AsycSlaveServeThread extends Thread {
	P2PBase p2PBase;
	ClientVo cv;
	AsycMaster asycMaster;
	int index;
	
	public AsycSlaveServeThread(int index, ClientVo cv, AsycMaster asycMaster) {
		super();
		this.index = index;
		this.cv = cv;
		this.asycMaster = asycMaster;
		p2PBase = new P2PBase(cv.getSocket(), cv.getOos(), 
				cv.getOis());
	}

	@Override
	public void run() {
		while (true) {
//			System.out.println("thread is ready to collect inf from client");
			Object sub = p2PBase.receiveObj();
			Object err = p2PBase.receiveObj();
			synchronized (asycMaster) {
				System.out.println("Prepare to merge client "+index
						+", "+this.p2PBase.socket);	
				Object [] subs = new Object [] {sub};
				if (sub == null) {
					System.out.println("Its sub uploaded is null..");
				}
				if (asycMaster.isCltSrvSameMode(subs)) {
					asycMaster.mergeSubject(subs);
					asycMaster.caculateErrorLastTime(new Object [] {err});
					Error error = (Error) err; 
					double avgErr = error.getErr()/(double)error.getCnt();
					if (error.isReady()) {
						avgErr = error.getErr();
					}					
					System.out.println("Prepare to merge client "+index
							+", run "+error.getCnt()+" samples, with avg error "+avgErr);	
				}				
				Object obj = asycMaster.getDistributeSubject();
				p2PBase.sendObj(obj);
				if (asycMaster.done) {
					try {
						asycMaster.testOnMaster();
					} catch (Exception e) {
						e.printStackTrace();
					}
					break;
				}				
			}
//			try {
//				this.sleep(0);
//			} catch (InterruptedException e) {
//				e.printStackTrace();
//			}
		}
	}
	

}
