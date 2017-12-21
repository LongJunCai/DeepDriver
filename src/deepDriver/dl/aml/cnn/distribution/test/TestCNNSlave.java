package deepDriver.dl.aml.cnn.distribution.test;

import deepDriver.dl.aml.cnn.distribution.CNNSlave;
import deepDriver.dl.aml.distribution.DistributionEnvCfg;
import deepDriver.dl.aml.distribution.P2PServer;

public class TestCNNSlave {
	
	public static void main(String[] args) throws Exception {
		String host = "127.0.0.1";
		if (args != null && args.length >= 1) { 
			host = args[0];
		} 	
		System.out.println("Connet to Server: "+host);
		DistributionEnvCfg.getCfg().set(P2PServer.KEY_SRV_HOST, host);
		DistributionEnvCfg.getCfg().set(P2PServer.KEY_SRV_PORT, 8034);
		
		CNNSlave cnnSlave = new CNNSlave();
		cnnSlave.train();
	}

}
