package deepDriver.dl.aml.lstm.distribution;


import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMWwUpdater;
import deepDriver.dl.aml.lstm.data.CfgDataTransfer;
import deepDriver.dl.aml.lstm.data.LSTMCfgData;

public class Seq2SeqSlaveV3 extends Seq2SeqSlave {

	public Seq2SeqSlaveV3(Seq2SeqLSTMBoostrapper boot) {
		super(boot);
	}
	
	LSTMCfgData cfgFromSrv;
	CfgDataTransfer cfgDataTransfer = new CfgDataTransfer();
	public void setSubject(Object obj) {
		cfgFromSrv = (LSTMCfgData) obj;
		testQ = Seq2SeqMasterV2.isQMode(cfgFromSrv);
		System.out.println("Subject is from server, " +
				"in Q mode? "+ testQ+", with round"+cfgFromSrv.getLoop());
		seq2SeqLSTM.getCfg().setTestQ(testQ);
		if (testQ) {
			cfgDataTransfer.copyData2Cfg(cfgFromSrv, seq2SeqLSTM.getCfg().getQlSTMConfigurator().getLayers());
		} else {
			cfgDataTransfer.copyData2Cfg(cfgFromSrv, seq2SeqLSTM.getCfg().getAlSTMConfigurator().getLayers());
		}		
		System.out.println("Already fresh Ww from server" );
	}	
	
	LSTMWwUpdater wWchecker = new LSTMWwUpdater(true, true);
	public void testWw(LSTMConfigurator fcfg) {
		System.out.println("Prepare to check Wws");
		wWchecker.updatewWs(fcfg, seq2SeqLSTM.getCfg().getQlSTMConfigurator());
		System.out.println("Done to check Wws");
	}

	@Override
	public Object getLocalSubject() {
		System.out.println("Prepare the local cfg.." );
		if (seq2SeqLSTM.getCfg().isTestQ()) {
			cfgFromSrv = cfgDataTransfer.loadCfg(
					seq2SeqLSTM.getCfg().getQlSTMConfigurator().
					getLayers());
		} else {
			cfgFromSrv = cfgDataTransfer.loadCfg(
					seq2SeqLSTM.getCfg().getAlSTMConfigurator().
					getLayers());
		}		
		System.out.println("Get the local cfg ready" );
		return cfgFromSrv;
	}

}
