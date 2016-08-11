package deepDriver.dl.aml.lstm.distribution.test;

import deepDriver.dl.aml.lstm.distribution.Seq2SeqAsycSlaveV6;

public class TestSeq2SeqAsycSlave {
	public static void main(String[] args) throws Exception {
//		Seq2SeqSlaveV2 slave = new Seq2SeqSlaveV2(new Seq2SeqLSTMSetup());
		Seq2SeqAsycSlaveV6 slave = new Seq2SeqAsycSlaveV6(new Seq2SeqLSTMSetup());
		slave.train();
	}
	
}
