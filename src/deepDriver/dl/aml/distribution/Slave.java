package deepDriver.dl.aml.distribution;

public abstract class Slave {
	protected P2PClient talkClient = new P2PClient();

	String currentCommand;
	public void train() throws Exception {
		talkClient.setup();
		talkClient.responseReady();
		while(true) {			
			String command = talkClient.receiveCommand();
			System.out.println("Receive command from server: "+command);
			if (command == null) {
				System.out.println("EORROR OCCURED ON SERVER, EXIT");
				break;
			}
			else if (Master.TaskCommand.equals(command)) {
				setTask(talkClient.receiveObj());
			} else if (Master.SubjectCommand.equals(command)) {
				setSubject(talkClient.receiveObj());
			} else if (Master.TrainCommand.equals(command)) {
				//afraid it may be time out...
				trainLocal();
			} else if (Master.CollectSubjectCommand.equals(command)) {
				talkClient.sendObj(getLocalSubject());
			} else if (Master.CollectErrorCommand.equals(command)) {				
				talkClient.sendObj(getError());
			} 
			currentCommand = command;
		}
	}
	
	public abstract void setTask(Object obj) throws Exception;
	
	public abstract void trainLocal() throws Exception;
	
	public abstract Error getError();
	
	public abstract void setSubject(Object obj);
	
	public abstract Object getLocalSubject();
	
}
