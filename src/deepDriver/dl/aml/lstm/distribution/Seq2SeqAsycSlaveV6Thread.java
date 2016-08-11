package deepDriver.dl.aml.lstm.distribution;

public class Seq2SeqAsycSlaveV6Thread extends Thread {
	Seq2SeqAsycSlaveV6 slave;
	
	public Seq2SeqAsycSlaveV6Thread(Seq2SeqAsycSlaveV6 slave) {
		super();
		this.slave = slave;
	}

	@Override
	public void run() {
		try {
			slave.train();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
