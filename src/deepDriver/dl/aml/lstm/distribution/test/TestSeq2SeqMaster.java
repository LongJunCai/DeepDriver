package deepDriver.dl.aml.lstm.distribution.test;

import java.io.File;


import deepDriver.dl.aml.distribution.DistributionEnvCfg;
import deepDriver.dl.aml.lstm.distribution.Seq2SeqMasterV5;

public class TestSeq2SeqMaster {
	
	public static void main(String[] args) throws Exception {
		DistributionEnvCfg.getCfg(). set(Seq2SeqLSTMSetup.KEY_FS_ROOT, args[0]);
		DistributionEnvCfg.getCfg(). set(Seq2SeqLSTMSetup.KEY_TEST_FILE, args[1]);
		Seq2SeqMasterV5 master = null;
		if (args.length > 3) {
			System.out.println("There are params passed in.");
			String sf = System.getProperty("user.dir");
			File mf = new File(sf, "data/"+args[2]); 
			if (mf.exists()) {
				master = new Seq2SeqMasterV5(
					4, new Seq2SeqLSTMSetup(), mf.getAbsolutePath());
			} else {
				master = new Seq2SeqMasterV5(
						4, new Seq2SeqLSTMSetup(), null);
			}			
		} else {
			master = new Seq2SeqMasterV5(
					4, new Seq2SeqLSTMSetup());
		}
		master.train();
	}

}
