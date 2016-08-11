package deepDriver.dl.aml.lstm.distribution;


import deepDriver.dl.aml.distribution.Error;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.data.CfgDataTransfer;
import deepDriver.dl.aml.lstm.data.LSTMCfgData;

public class Seq2SeqMasterV3 extends Seq2SeqMaster {
	public Seq2SeqMasterV3(int clientsNum, Seq2SeqLSTMBoostrapper boot) throws Exception {
		super(clientsNum, boot);
	}
	
	public static int QType = 1;
	public static int AType = 2;	
	LSTMCfgData cfgFromClient;
	
	public static boolean isQMode(LSTMConfigurator lstmCfgFromClient) {
		if (lstmCfgFromClient.getType() == QType) {
			return true;
		}
		return false;
	}
	
	CfgDataTransfer cfgDataTransfer = new CfgDataTransfer();
	
	public Object getDistributeSubject() {
		System.out.println("Prepare srv cfg to clients"+srvQ2QLSTM.getCfg().getLoop());
		if (srvQ2QLSTM.getCfg().isTestQ()) {
			cfgFromClient = cfgDataTransfer.loadCfg(
					srvQ2QLSTM.getCfg().getQlSTMConfigurator().
					getLayers());
			cfgFromClient.setType(QType);
		} else {
			cfgFromClient = cfgDataTransfer.loadCfg(
					srvQ2QLSTM.getCfg().getAlSTMConfigurator().
					getLayers());
			cfgFromClient.setType(AType);
		}	
		cfgFromClient.setLoop(srvQ2QLSTM.getCfg().getLoop());

		System.out.println("Ready for srv cfg to clients"+cfgFromClient.getLoop());
		return cfgFromClient;
	}
	
	public void mergeSubject(Object[] objs) {
		System.out.println("Prepare to merge Ww from clients");
		LSTMCfgData[] cfgs = new LSTMCfgData[objs.length];
		for (int i = 0; i < cfgs.length; i++) {
			cfgs[i] = (LSTMCfgData) objs[i]; 
		}
		srvQ2QLSTM.getCfg().setLoop(srvQ2QLSTM.getCfg().getLoop() + 1); 
		if (srvQ2QLSTM.getCfg().isTestQ()) {
			this.cfgDataTransfer.mergeDeltaWw2Cfg(cfgs, srvQ2QLSTM.getCfg().
					getQlSTMConfigurator().getLayers());
			lSTMWwFresher.freshwWs(this.srvQ2QLSTM.getCfg().
				getQlSTMConfigurator());
		} else {
			this.cfgDataTransfer.mergeDeltaWw2Cfg(cfgs, srvQ2QLSTM.getCfg().
					getAlSTMConfigurator().getLayers());
			lSTMWwFresher.freshwWs(this.srvQ2QLSTM.getCfg().
				getAlSTMConfigurator());
		}	
				
		System.out.println("Done with Ww updates");
		
	}
	
	public double caculateErrorLastTime(Object[] objs) {
		double err = 0;
		double cnt = 0;		
		for (int i = 0; i < objs.length; i++) {
			Error ce = (Error)objs[i];
			if (!ce.isReady()) {
				return 0;
			}
			err = err + ce.getErr();
			cnt = cnt + 1;//+ ce.getCnt();
		}
		err = err/cnt;
		if (srvQ2QLSTM.getCfg().isTestQ()) {
			if (err <= srvQ2QLSTM.getCfg().getQlSTMConfigurator().getAccuracy()) {
				srvQ2QLSTM.getCfg().setTestQ(false);
				System.out.println("Switch to A mode, and reset the cfg");
				cfgFromClient = null;
			}
			System.out.println(cnt+" is tested Q with error "+ err);
		} else {
			if (err <= srvQ2QLSTM.getCfg().getAlSTMConfigurator().getAccuracy()) {
				this.done = true;
				System.out.println("Complete training");
			}
			System.out.println(cnt+" is tested A with error "+ err);
		}		
		return err;
	}
}
