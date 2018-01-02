package deepDriver.dl.aml.common.distribution;

import deepDriver.dl.aml.distribution.Error;
import deepDriver.dl.aml.distribution.Slave;

public class CommonSlave extends Slave {
	public static String CMODEL_SLAVE = "-CMODEL_SLAVE";
	Slave ms = null;
	public void handleOthers(String command) throws Exception {
		if (command.startsWith(CMODEL_SLAVE)) {
			String clazz = command.substring(CMODEL_SLAVE.length() + 1); 
			System.out.println("Prepare to run "+clazz);
			ms = (Slave) Class.forName(clazz).newInstance();
		} else {
			ms.handleOthers(command);
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
