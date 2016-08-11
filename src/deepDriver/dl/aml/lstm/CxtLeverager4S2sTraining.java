package deepDriver.dl.aml.lstm;

import java.io.Serializable;

public class CxtLeverager4S2sTraining implements IPreCxtProvider, ICxtConsumer, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public CxtLeverager4S2sTraining(LSTM qLSTM, LSTM aLSTM) {
		super();
		this.qLSTM = qLSTM;
		this.aLSTM = aLSTM;
	}

	Context currentContext = null; 
 	public void reset() {
	}
	
	public boolean hasNext() {
		if(currentContext != null) {
			return true;
		} 
		return false;
	}
	
	public Context next() {
		Context ctx = currentContext;
		currentContext = null;
		return ctx;
	}

	@Override
	public void addContext(Context cxt) {
		currentContext = cxt;
	}

	@Override
	public void complete() {
	}

	@Override
	public boolean isCompleted() {
		return false;
	}

	LSTM qLSTM;
	LSTM aLSTM;
	Object requireObj;
	@Override
	public void require(Object obj) {
		Object pos = aLSTM.is.getPos();
		qLSTM.is.next(pos);
		qLSTM.test(qLSTM.is.getSampleTT(), qLSTM.is.getTarget());
//		qLSTM.test(sample, targets);
	}

}
