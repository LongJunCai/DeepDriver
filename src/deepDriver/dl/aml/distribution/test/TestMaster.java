package deepDriver.dl.aml.distribution.test;


import deepDriver.dl.aml.distribution.Fs;
import deepDriver.dl.aml.distribution.ITask;
import deepDriver.dl.aml.distribution.Master;
import deepDriver.dl.aml.distribution.test.cl.Employee;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.Seq2SeqLSTMConfigurator;
import deepDriver.dl.aml.lstm.data.CfgDataCleaner;
import deepDriver.dl.aml.lstm.data.LSTMCfgData;
import deepDriver.dl.aml.lstm.distribution.SimpleTask;

public class TestMaster extends Master {

	@Override
	public void testOnMaster() throws Exception {
	}

	@Override
	public int getClientsNum() {
		return 2;
	}

	@Override
	public Object[] splitTasks() {
		return new ITask[]{new SimpleTask(1, 1, 1)};
	}

	HelloCnt hello = new HelloCnt();
	@Override
	public Object getDistributeSubject() {
//		hello.i ++;
		HelloWrapper w = new HelloWrapper();
//		HelloCnt hc = new HelloCnt();
//		hc.i = hello.i;
		w.helloCnt = hello;
		return w;
	}

	@Override
	public double caculateErrorLastTime(Object[] objs) { 
		return 0;
	}

	@Override
	public void mergeSubject(Object[] objs) { 

	}
	
	@Override
	public void train() throws Exception { 
		talkServer.setup(getClientsNum());
//		talkServer.collectState();
		
		while (true) {	
			cnt ++;
			seq2seq3();
//			hello();
//			test();
		}
	}
	public void test() throws Exception {
		double [][][] aa = new double[1000][][];
		for (int i = 0; i < aa.length; i++) {
			aa[i] = new double[6][];
			for (int j = 0; j < aa[i].length; j++) {
				aa[i][j] = new double[1000];
				for (int j2 = 0; j2 < aa.length; j2++) {
					aa[i][j][j2] = 0.1;
				}
			}
		}
		aa[0][0][0] = aa[0][0][0] + cnt;
		talkServer.distributeObject(aa);
		System.out.println("Send seq2seq to client"+cnt);
//		Object [] objs = talkServer.collectObjs();
	}
	
	LSTMCfgData cData;
	public void seq2seq3() throws Exception {
//		if (cnt > 10) {
//			talkServer.close();
//		}
		if (cData == null) {
			cData = (LSTMCfgData) Fs.
					readObjFromFile
					("D:\\6.workspace\\ANN\\t3\\seq2seqCfg1448765458973");
			
//			Fs.writeObj2FileWithTs("D:\\6.workspace\\ANN\\qCfg", seq2seqCfg.getQlSTMConfigurator());
//			Fs.writeObj2FileWithTs("D:\\6.workspace\\ANN\\aCfg", seq2seqCfg.getAlSTMConfigurator());
		}  
		cData.getCfg()[0][0][0] = cnt + 1;
		System.out.println("The data length is: "+cData.getCfg().length);
		System.out.println(cData.getCfg()[0][0][0]);
		talkServer.distributeObject(cData.getCfg());
		System.out.println("Send seq2seq to client");
		Object [] objs = talkServer.collectObjs();
		System.out.println("Collect seq2seq from client");
		double [][][][] seqs = new double[objs.length][][][];
		for (int i = 0; i < seqs.length; i++) {
			seqs[i] = (double[][][]) objs[i];
//			if (i > 0) {
//				CfgDataCleaner.clean(seqs[i]);
//				seqs[i] = null;
//				objs[i] = null;
//			}
//			seqs[i] = null;
		}
//		CfgDataCleaner.clean(cData.getCfg());
		cData.setCfg((double [][][]) objs[0]);
//		objs[0] = null;
	}
	
	public void seq2seq2() throws Exception {
//		if (cnt > 10) {
//			talkServer.close();
//		}
		if (seq2seqCfg == null) {
			seq2seqCfg = (Seq2SeqLSTMConfigurator) Fs.
					readObjFromFile
					("D:\\6.workspace\\ANN\\seq2seqCfg1448614350537");
			
//			Fs.writeObj2FileWithTs("D:\\6.workspace\\ANN\\qCfg", seq2seqCfg.getQlSTMConfigurator());
//			Fs.writeObj2FileWithTs("D:\\6.workspace\\ANN\\aCfg", seq2seqCfg.getAlSTMConfigurator());
		}
		talkServer.distributeObject(seq2seqCfg.
				getQlSTMConfigurator());
		System.out.println("Send seq2seq to client");
		Object [] objs = talkServer.collectObjs();
		System.out.println("Collect seq2seq from client");
		LSTMConfigurator [] seqs = new LSTMConfigurator[objs.length];
		for (int i = 0; i < seqs.length; i++) {
			seqs[i] = (LSTMConfigurator) objs[i];
		}
		seq2seqCfg.setQlSTMConfigurator(seqs[0]);
	}
	
	Seq2SeqLSTMConfigurator seq2seqCfg = null;
	public void seq2seq() throws Exception {
//		if (cnt > 10) {
//			talkServer.close();
//		}
		if (seq2seqCfg == null) {
			seq2seqCfg = (Seq2SeqLSTMConfigurator) Fs.
					readObjFromFile
					("D:\\6.workspace\\ANN\\seq2seqCfg1448614350537");
			
//			Fs.writeObj2FileWithTs("D:\\6.workspace\\ANN\\qCfg", seq2seqCfg.getQlSTMConfigurator());
//			Fs.writeObj2FileWithTs("D:\\6.workspace\\ANN\\aCfg", seq2seqCfg.getAlSTMConfigurator());
		}
		talkServer.distributeObject(seq2seqCfg.getQlSTMConfigurator());
		System.out.println("Send seq2seq to client");
		Object [] objs = talkServer.collectObjs();
		System.out.println("Collect seq2seq from client");
		LSTMConfigurator [] seqs = new LSTMConfigurator[objs.length];
		for (int i = 0; i < seqs.length; i++) {
			seqs[i] = (LSTMConfigurator) objs[i];
		}
		seq2seqCfg.setQlSTMConfigurator(seqs[0]);
	}
	
	int cnt = 0;
	public void employee() throws Exception {
		Employee employee = (Employee )talkServer.collectObjs()[0];

         employee .setEmployeeNumber(256+cnt);
         employee .setEmployeeName("John");

         talkServer.distributeObject(employee);
         
         employee = (Employee )talkServer.collectObjs()[0];
         System.out.println("employeeNumber= "
                 + employee .getEmployeeNumber());
 System.out.println("employeeName= "
                 + employee .getEmployeeName());
	}
	
	public void hello() throws Exception {
		hello.setI(hello.getI() + 2);
		int [] k = hello.getK();
		for (int i = 0; i < k.length; i++) {
			k[i] = i + 1;
			System.out.println("k"+"["+i+"]"+k[i]);
		}
			talkServer.distributeObject(hello);
			System.out.println(hello.getI()+","+hello.getJ());
			HelloCnt hc = (HelloCnt) talkServer.collectObjs()[0];
			System.out.println(hc.getI()+","+hc.getJ());
	}
	
	public static void main(String[] args) throws Exception {
		TestMaster master = new TestMaster();
		master.train();
	}

}
