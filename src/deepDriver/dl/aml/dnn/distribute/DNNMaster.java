package deepDriver.dl.aml.dnn.distribute;

import java.io.Serializable;

import deepDriver.dl.aml.ann.InputParameters;
import deepDriver.dl.aml.distribution.ResourceMaster;
import deepDriver.dl.aml.dnn.DNN;

public class DNNMaster implements Serializable {
	public static String  Task_caculateHiddenInputs = "Task_caculateHiddenInputs";
	public static String  Task_ExpendLayer = "Task_ExpendLayer";
	public static String  Task_Pre_Training = "Task_Pre_Training";
	public static String  Task_FineTuning = "Task_FineTuning";
	
	public boolean isSetup() {
		return ResourceMaster.getInstance().isSetup();
	}
	
	private static final long serialVersionUID = 1L;
	public void distributeTasks(DNN dnn, InputParameters parameters) throws Exception {
		if (!isSetup()) {
			return;
		}
		ResourceMaster rm = ResourceMaster.getInstance();
		Object [] tasks = DNNDistUtils.splitTasks(dnn.getResults(parameters), rm.getClientsNum(), parameters);
		rm.distributeTasks(tasks);
		rm.distributeSubject(dnn);
	}
	
	public void distributeDNN(DNN dnn) throws Exception {
		if (!isSetup()) {
			return;
		}
		ResourceMaster rm = ResourceMaster.getInstance();		
		rm.distributeSubject(dnn);
	}
	
	public void caculateHiddenInputs() throws Exception {
		if (!isSetup()) {
			return;
		}
		ResourceMaster rm = ResourceMaster.getInstance();
//		String [] tasks = new String[rm.getClientsNum()];
//		for (int i = 0; i < tasks.length; i++) {
//			tasks[i] = Task_caculateHiddenInputs;
//		}
//		rm.distributeTasks(tasks);
		rm.distributeCommand(Task_caculateHiddenInputs);
	}
	
	public void expendLayer() throws Exception {
		if (!isSetup()) {
			return;
		}
		ResourceMaster rm = ResourceMaster.getInstance(); 
		rm.distributeCommand(Task_ExpendLayer);
	}
	
	public void preTraining() throws Exception {
		if (!isSetup()) {
			return;
		}
		ResourceMaster rm = ResourceMaster.getInstance(); 
		rm.distributeCommand(Task_Pre_Training);
	}
	
	public void distFineTuning() throws Exception {
		if (!isSetup()) {
			return;
		}
		ResourceMaster rm = ResourceMaster.getInstance(); 
		rm.distributeCommand(Task_FineTuning);
	}

}
