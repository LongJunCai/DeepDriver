package deepDriver.dl.aml.distribution;

public abstract class AsycSlave {
	protected P2PClient talkClient = new P2PClient();

	String currentCommand;
	public void train() throws Exception {
		prepareData(false);
		talkClient.setup();
		talkClient.responseReady();
		
		String command = talkClient.receiveCommand();
		System.out.println("Receive command from server: "+command);
		setTask(talkClient.receiveObj());
		setSubject(talkClient.receiveObj());
		if (Master.TaskCommand.equals(command)) {			
		} else if (Master.SubjectCommand.equals(command)) {			
		}
		while(true) {
			long l1 = System.currentTimeMillis();
			trainLocal();
			long l2 = System.currentTimeMillis();
			System.out.println("Training time cost: "+(l2 - l1));
			talkClient.sendObj(getLocalSubject());
			talkClient.sendObj(getError());
			setSubject(talkClient.receiveObj());
			long l3 = System.currentTimeMillis();
			System.out.println("Switch data cost: "+(l3 - l2));
			System.out.println("The threads num per CPU should be: "+(l3 - l2)/(l2 - l1));
		}
	}
	
	public abstract void prepareData(boolean isServer) throws Exception;
	
	public abstract void setTask(Object obj) throws Exception;
	
	public abstract void trainLocal() throws Exception;
	
	public abstract Error getError();
	
	public abstract void setSubject(Object obj);
	
	public abstract Object getLocalSubject();
	
}
