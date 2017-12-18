package deepDriver.dl.aml.distribution;

public abstract class Master {
	protected P2PServer talkServer = new P2PServer();
	protected boolean done = false;
	public static String TrainCommand = "-c train";
	public static String TaskCommand = "-c task";
	public static String SubjectCommand = "-c subject";
	public static String CollectErrorCommand = "-c collectError";
	public static String CollectSubjectCommand = "-c collectSubject";
	int cnt = 0;
	
	boolean setup = false;
	public void setup() throws Exception {
		talkServer.setup(getClientsNum());
		talkServer.collectState();
		setup = true;
	}
	
	public void train() throws Exception {
		if (!setup) {
			setup();
		}
		while (true) {			
			talkServer.distributeCommand(TaskCommand);			
			talkServer.distributeObjects(splitTasks());
			talkServer.distributeCommand(SubjectCommand);	
			Object obj = getDistributeSubject();
			talkServer.distributeObject(obj);
//			Fs.writeObj2FileWithTs("D:\\6.workspace\\ANN\\seq2seqCfg", obj);
			talkServer.distributeCommand(TrainCommand);
			talkServer.distributeCommand(CollectSubjectCommand);
			mergeSubject(talkServer.collectObjs());
			testOnMaster();
			talkServer.distributeCommand(CollectErrorCommand);
			caculateErrorLastTime(talkServer.collectObjs());
			if (done) {
				testOnMaster();
				break;
			}
		}		
	}
	
//	public int setup() {		
//		return talkServer.getClients().size();
//	}
	public abstract void testOnMaster() throws Exception;
	
	public abstract int getClientsNum();
	
	public abstract Object [] splitTasks();
	
	public abstract Object getDistributeSubject(); 
//	
//	public abstract void trainOnSlave();
	
	public abstract double caculateErrorLastTime(Object [] objs);
	
	public abstract void mergeSubject(Object [] objs);

}
