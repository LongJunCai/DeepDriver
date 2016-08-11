package deepDriver.dl.aml.distribution;

import java.util.ArrayList;
import java.util.List;

public abstract class AsycMaster {
	protected P2PServer talkServer = new P2PServer();
	protected boolean done = false;
	public static String TrainCommand = "-c train";
	public static String TaskCommand = "-c task";
	public static String SubjectCommand = "-c subject";
	public static String CollectErrorCommand = "-c collectError";
	public static String CollectSubjectCommand = "-c collectSubject";
	int cnt = 0;

	List<AsycSlaveServeThread> slaveThreads = new ArrayList<AsycSlaveServeThread>();
	public void train() throws Exception {
		talkServer.setup(getClientsNum());
		talkServer.collectState();
		
		List<ClientVo> clients = talkServer.getClients();
		for (int i = 0; i < getClientsNum(); i++) {
			slaveThreads.add(new AsycSlaveServeThread(i, clients.get(i), this));
		}
		
		talkServer.distributeCommand(TaskCommand);			
		talkServer.distributeObjects(splitTasks());
		Object obj = getDistributeSubject();
		talkServer.distributeObject(obj);
		
		for (int i = 0; i < slaveThreads.size(); i++) {
			slaveThreads.get(i).start();
			System.out.println(i+" client thread started.");
		}
		
		for (int i = 0; i < slaveThreads.size(); i++) {
			slaveThreads.get(i).join();
		}
		System.out.println("Master exit");
	}		

	public abstract void testOnMaster() throws Exception;
	
	public abstract int getClientsNum();
	
	public abstract Object [] splitTasks();
	
	public abstract Object getDistributeSubject(); 
	
	public abstract double caculateErrorLastTime(Object [] objs);
	
	public abstract void mergeSubject(Object [] objs);
	
	public abstract boolean isCltSrvSameMode(Object [] objs);
	
}
