package deepDriver.dl.aml.lstm.distribution;

import deepDriver.dl.aml.lstm.ICell;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMLayer;

public class Verifier {
	
	//set the 100 cells Wws 1;
	int cellIndex = 100;
	double deltaWw = 1;
	double wW = 2;
	
	int clientsNum = 4;
	
	boolean verify = true;
	
	public Verifier(boolean verify) {
		super();
		this.verify = verify;
	}

	//cfg on the server before distribution
	public void cfgDistributeWws(LSTMConfigurator cfg) {
		if (!verify) {
			return ;
		}
		ICell cell = getTestCell(cfg);
		setWws(cell.getwWs(), wW);	
		System.out.println("Distribute test Wws");
		setWws(cell.getDeltaWWs(), deltaWw);	
		System.out.println("Distribute test DeltaWws");
	}
	
	
	
	//verify on the client after copy
	public void verifyDistributeWws(LSTMConfigurator cfg) {
		if (!verify) {
			return ;
		}
		ICell cell = getTestCell(cfg);
		if (test(cell.getwWs(), wW)) {
			System.out.println("The Wws are distributed successfully.");
		} else {
			System.out.println("The Wws are distributed failed.");
		}
		if (test(cell.getDeltaWWs(), deltaWw)) {
			System.out.println("The DeltaWws are distributed successfully.");
		} else {
			System.out.println("The DeltaWws are distributed failed.");
		}
		
	}
	
	//cfg on the clients
	public void cfgClientWws(LSTMConfigurator cfg) {
		cfgDistributeWws(cfg);
	}
	
	//verify on the server after merge
	public void verifyMergeWws(LSTMConfigurator cfg) {
		if (!verify) {
			return ;
		}
		ICell cell = getTestCell(cfg);
		if (test(cell.getwWs(), wW, deltaWw * clientsNum)) {
			System.out.println("The Wws are merged successfully.");
		} else {
			System.out.println("The Wws are merged failed.");
		}
		if (test(cell.getDeltaWWs(), deltaWw * clientsNum)) {
			System.out.println("The DeltaWws are merged successfully.");
		} else {
			System.out.println("The DeltaWws are merged failed.");
		}
	}
	
	public boolean test(double [] wWs, double value, double delta) {
		if (wWs == null) {
			return true;
		}
		for (int i = 0; i < wWs.length; i++) {
			if (wWs[i] != value + delta) {
				return false;
			}
		}
		return true;
	}
	
	public boolean test(double [] wWs, double value) {
		if (wWs == null) {
			return true;
		}
		for (int i = 0; i < wWs.length; i++) {
			if (wWs[i] != value) {
				return false;
			}
		}
		return true;
	}

	
	private void setWws(double [] wWs, double value) {
		if (wWs == null) {
			return ;
		}
		for (int i = 0; i < wWs.length; i++) {
			wWs[i] = value; 
		}
	}
	
	private ICell getTestCell(LSTMConfigurator cfg) {
		LSTMLayer layer = (LSTMLayer) cfg.getLayers()[1];
		return layer.getBlocks()[0].getCells()[100];
	}

}
