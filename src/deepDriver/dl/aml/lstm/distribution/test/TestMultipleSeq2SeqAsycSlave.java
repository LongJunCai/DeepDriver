package deepDriver.dl.aml.lstm.distribution.test;

import java.util.ArrayList;
import java.util.List;

import deepDriver.dl.aml.lstm.distribution.Seq2SeqAsycSlaveV6;
import deepDriver.dl.aml.lstm.distribution.Seq2SeqAsycSlaveV6Thread;

public class TestMultipleSeq2SeqAsycSlave {
	public static void main(String[] args) throws Exception {
//		Seq2SeqSlaveV2 slave = new Seq2SeqSlaveV2(new Seq2SeqLSTMSetup());
		int sn = TestSeq2SeqAsycMaster.CLIENT_NUM;
		List<Seq2SeqAsycSlaveV6Thread> ths = 
				new ArrayList<Seq2SeqAsycSlaveV6Thread>();
		for (int i = 0; i < sn; i++) {
			ths.add(new Seq2SeqAsycSlaveV6Thread(new Seq2SeqAsycSlaveV6(new Seq2SeqLSTMSetup())));
		}
		for (int i = 0; i < sn; i++) {
			ths.get(i).start();
		}
		for (int i = 0; i < sn; i++) {
			ths.get(i).join();
		}
	}
	
}
