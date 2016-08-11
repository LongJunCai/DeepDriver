package deepDriver.dl.aml.lstm.distribution.test;


import deepDriver.dl.aml.distribution.DistributionEnvCfg;
import deepDriver.dl.aml.distribution.P2PServer;
import deepDriver.dl.aml.lstm.distribution.Seq2SeqAsycSlaveV6;

public class TestQAWorker {
	public static void main(String[] args) throws Exception {
		DistributionEnvCfg.getCfg(). set(Seq2SeqLSTMSetup.KEY_FS_ROOT, args[0]);
		DistributionEnvCfg.getCfg(). set(Seq2SeqLSTMSetup.KEY_TEST_FILE, args[1]);
		DistributionEnvCfg.getCfg(). set(P2PServer.KEY_SRV_HOST, args[2]);
//		DistributionEnvCfg.getCfg(). set(P2PServer., "10.1.242.48");
		
//		Seq2SeqSlaveV2 slave = new Seq2SeqSlaveV2(new Seq2SeqLSTMSetup());
		Seq2SeqAsycSlaveV6 slave = new Seq2SeqAsycSlaveV6(new Seq2SeqLSTMSetup());
		slave.train();
	}
	
}
