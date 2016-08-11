package deepDriver.dl.aml.lstm.test;


import deepDriver.dl.aml.distribution.DistributionEnvCfg;
import deepDriver.dl.aml.lstm.conversation.Seq2SeqConversation;

public class TestQa {
	
	public static void main(String[] args) throws Exception {
		Seq2SeqConversation converstion = new Seq2SeqConversation();
		String root = "D:\\6.workspace\\ANN\\lstm\\QaModel\\";
		Seq2SeqBabySysSetup s2s = new Seq2SeqBabySysSetup(); 
		s2s.setThreadsNum(4);
		DistributionEnvCfg.getCfg(). set(Seq2SeqBabySysSetup.KEY_FS_ROOT, root);
		DistributionEnvCfg.getCfg(). set(Seq2SeqBabySysSetup.KEY_TEST_FILE, 
//				"talk2016.txt");
				"talk2015_2016.txt");
//		DistributionEnvCfg.getCfg(). set(P2PServer.KEY_SRV_HOST, args[2]);
		
		/**converstion.load(s2s,  
				root+"qModel_v1466671795460_1.m" , root+"aModel_v1467890329260_0.m",//
				root+"v.m");**/
		
		converstion.load(s2s,  
				root+"qModel_v1516_1468479736787_0.m" , 
				root+"aModel_v1516_1468846020244_0.m",//
				root+"v_1516.m");
//		converstion.testQ("感谢");
		long l = System.currentTimeMillis();
		converstion.testQas("怎么我的是黄金套餐还要购买才能看电影?", 3, 41); //我账号密码是什么？我看视频一直缓冲为什么如何取消自动续费   你工号多少，我投诉你""
		System.out.println((System.currentTimeMillis() - l)+" costed.");
		
	}

}
