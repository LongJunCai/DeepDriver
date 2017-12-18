package deepDriver.dl.aml.distribution;

public class ResourceMaster {
	
	static ResourceMaster instance = new ResourceMaster();

	protected P2PServer talkServer = new P2PServer();
	protected boolean done = false;
	public static String TrainCommand = "-c train";
	public static String TaskCommand = "-c task";
	public static String SubjectCommand = "-c subject";
	public static String CollectErrorCommand = "-c collectError";
	public static String CollectSubjectCommand = "-c collectSubject";
	int cnt = 0;
	int clientsNum;
	
	boolean setup = false;
	
	public static ResourceMaster getInstance() {
		return instance;
	}

//	public static void setInstance(ResourceMaster instance) {
//		ResourceMaster.instance = instance;
//	}

	public void setup(int clientsNum) throws Exception {
		this.clientsNum = clientsNum;
		talkServer.setup(clientsNum);
		talkServer.collectState();
		setup = true;
	}
	
	Object [] errs;
	public void distributeCommand(String command) throws Exception {
		talkServer.distributeCommand(command); 
	}
	
	public void distributeTasks(Object [] tasks) throws Exception {
		if (tasks != null) {
			talkServer.distributeCommand(TaskCommand);
//			talkServer.distributeObjects(tasks);
			talkServer.distributeObjectsAsc(null, tasks);
		}
	}
	
	public void distributeSubject(Object cm) throws Exception {
		if (cm != null) {
			talkServer.distributeCommand(SubjectCommand); 
//			Fs.writeObj2FileWithTs("D:\\6.workspace\\ANN\\ANN.cfg", cm);
//			talkServer.distributeObject(cm);
			talkServer.distributeObjectsAsc(cm, null);
		}
	}
	
	public Object [] run(Object [] tasks, Object cm) throws Exception {
		distributeTasks(tasks);		
		distributeSubject(cm);		
		 
		talkServer.distributeCommand(TrainCommand);
		talkServer.distributeCommand(CollectSubjectCommand);
//		Object [] objs = talkServer.collectObjs(); 
		Object [] objs = talkServer.collectObjsAsc();
		talkServer.distributeCommand(CollectErrorCommand);
//		errs = talkServer.collectObjs(); 	
		errs = talkServer.collectObjsAsc();
		return objs;
	}

	public int getClientsNum() {
		return clientsNum;
	}

	public void setClientsNum(int clientsNum) {
		this.clientsNum = clientsNum;
	}

	public Object[] getErrs() {
		return errs;
	}

	public void setErrs(Object[] errs) {
		this.errs = errs;
	}

	public boolean isSetup() {
		return setup;
	}

	public void setSetup(boolean setup) {
		this.setup = setup;
	}	
}
