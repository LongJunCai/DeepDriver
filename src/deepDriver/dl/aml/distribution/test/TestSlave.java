package deepDriver.dl.aml.distribution.test;


import deepDriver.dl.aml.distribution.Error;
import deepDriver.dl.aml.distribution.Slave;
import deepDriver.dl.aml.distribution.test.cl.Employee;
import deepDriver.dl.aml.lstm.data.CfgDataCleaner;

public class TestSlave extends Slave {

	@Override
	public void setTask(Object obj) throws Exception {
	}

	@Override
	public void trainLocal() throws Exception {
	}

	@Override
	public Error getError() {
		return new Error();
	}

	@Override
	public void setSubject(Object obj) {
		HelloWrapper h = (HelloWrapper) obj;
		System.out.println(h+","+h.helloCnt.getI());
	}

	@Override
	public Object getLocalSubject() {
		return new HelloCnt();
	}
	
	Employee joe = new Employee(150, "Joe");
	@Override
	public void train() throws Exception {
		talkClient.setup();
//		talkClient.responseReady();
		int cnt = 0;
		while(true) {
			cnt++ ;
			seq2seq();
//			hello();
//			test();
		}
	}
	
	public void test() throws Exception {
		double [][][] seq = (double [][][]) talkClient.receiveObj();
		System.out.println("Received seq from server: "+seq[0][0][0]);	

//		talkClient.sendObj(seq);
//		CfgDataCleaner.clean(seq);
		System.out.println("send to server: ");	
	}
	
	public void seq2seq() {
		double [][][] seq = (double [][][]) talkClient.receiveObj();
		System.out.println("Received seq from server: "+seq[0][0][0]);	

		talkClient.sendObj(seq);
//		CfgDataCleaner.clean(seq);
		System.out.println("send to server: ");	

	}
	
	public void employee() {
//		System.out.println("employeeNumber= "
//        + joe .getEmployeeNumber());
//System.out.println("employeeName= "
//        + joe .getEmployeeName());
////clientOutputStream.writeObject(joe);
////clientOutputStream.flush();
//talkClient.sendObj(joe);
//System.out.println(joe);
//joe = (Employee)talkClient.receiveObj();			
//System.out.println(joe);
//System.out.println("employeeNumber= "
//                + joe.getEmployeeNumber());
//System.out.println("employeeName= "
//                + joe.getEmployeeName());
//joe.setEmployeeName("XXXXXXX");
//joe.setEmployeeNumber(1+cnt);
////clientOutputStream.writeObject(joe2);
//talkClient.sendObj(joe);
	}
	
	public void hello() {
		HelloCnt hc = (HelloCnt) talkClient.receiveObj();
			System.out.println(hc+","+hc.getI());	
			int [] k = hc.getK();
			for (int i = 0; i < k.length; i++) {
				System.out.println("k"+"["+i+"]"+k[i]);
			}
//			hc.j++;
			hc.setJ(hc.getJ() + 1);
			talkClient.sendObj(hc);
			System.out.println(hc.getI()+","+hc.getJ());
	}
	
	public static void main(String[] args) throws Exception {
		TestSlave s = new TestSlave();
		s.train();
	}

}
