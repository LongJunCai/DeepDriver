package deepDriver.dl.aml.dnn.distribute.test;

import deepDriver.dl.aml.distribution.DistributionEnvCfg;
import deepDriver.dl.aml.distribution.P2PServer;
import deepDriver.dl.aml.dnn.distribute.DNNSlave;

public class DNNSlaveTest {
	
	public static void main(String[] args) throws Exception {
		String host = "127.0.0.1";
		if (args != null && args.length >= 1) { 
			host = args[0];
		} 	
		System.out.println("Connet to Server: "+host);
		DistributionEnvCfg.getCfg().set(P2PServer.KEY_SRV_HOST, host);
		DistributionEnvCfg.getCfg().set(P2PServer.KEY_SRV_PORT, 8034);
		
		DNNSlave ds = new DNNSlave();
		ds.train();
	}

}
