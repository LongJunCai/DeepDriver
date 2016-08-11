package deepDriver.dl.aml.lstm.distribution.test;

import java.io.File;

import deepDriver.dl.aml.lstm.distribution.Seq2SeqAsycMasterV6;

public class TestSeq2SeqAsycMaster {
	static int CLIENT_NUM = 4;
	
	public static void main(String[] args) throws Exception {
		Seq2SeqAsycMasterV6 master = null;
		int sn = CLIENT_NUM;
		if (args.length > 0) {
			System.out.println("There are params passed in.");
			String sf = System.getProperty("user.dir");
			File mf = new File(sf, "data/"+args[0]); 
			if (mf.exists()) {
				master = new Seq2SeqAsycMasterV6(
					sn, new Seq2SeqLSTMSetup(), mf.getAbsolutePath());
			} else {
				master = new Seq2SeqAsycMasterV6(
						sn, new Seq2SeqLSTMSetup(), null);
			}			
		} else {
			master = new Seq2SeqAsycMasterV6(
					sn, new Seq2SeqLSTMSetup());
		}
		master.train();
	}

}
