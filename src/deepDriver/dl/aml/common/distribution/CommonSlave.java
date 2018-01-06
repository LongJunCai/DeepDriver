package deepDriver.dl.aml.common.distribution;

import deepDriver.dl.aml.distribution.Error;
import deepDriver.dl.aml.distribution.Slave;

public class CommonSlave extends Slave {
	public static String CMODEL_SLAVE = "-CMODEL_SLAVE";
	public static String CTASKPIECE = "-CTASKPIECE";
	
	Slave ms = null;
	Linkable root;
	Linkable current;
	public void handleOthers(String command) throws Exception {
		if (command.startsWith(CMODEL_SLAVE)) {
			String clazz = command.substring(CMODEL_SLAVE.length() + 1); 
			System.out.println("Prepare to run "+clazz);
			ms = (Slave) Class.forName(clazz).newInstance();
		} else if (command.startsWith(CTASKPIECE)) {
//			String clazz = command.substring(CTASKPIECE.length() + 1); 
//			System.out.println("Prepare to run "+clazz);
			Object obj = talkClient.receiveObj();
			if (obj instanceof String) {
				System.out.println("WTf: "+obj);
			}
			Linkable linkable1 = (Linkable) obj;
			if (root == null) {
				root = linkable1;
				current = root;
				if (ms != null) {
					ms.setTask(root);
				}
			} else {
				current.setNext(linkable1);
				current = linkable1;
			}
		} else {
			if (ms != null) {
				ms.handleOthers(command);
			}			
		}
	}

	@Override
	public void setTask(Object obj) throws Exception {
		ms.setTask(obj);
	}

	@Override
	public void trainLocal() throws Exception {
		ms.trainLocal();
	}

	@Override
	public Error getError() {
		return ms.getError();
	}

	@Override
	public void setSubject(Object obj) {
		ms.setSubject(obj);
	}

	@Override
	public Object getLocalSubject() {
		return ms.getLocalSubject();
	}

}
