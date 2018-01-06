package deepDriver.dl.aml.lstm.distribution;

import deepDriver.dl.aml.distribution.Error;
import deepDriver.dl.aml.distribution.Slave;
import deepDriver.dl.aml.lstm.IStream;
import deepDriver.dl.aml.lstm.LSTM;
import deepDriver.dl.aml.lstm.LSTMWwArrayTranslator;

public class LSTMSlave extends Slave {
	
	LSTM lstm;
	static int mb = 1024;
	IStream is;
	@Override
	public void setTask(Object obj) throws Exception {
		is = (IStream) obj;
	}

	double err = 0;
	Error error = new Error();
	@Override
	public void trainLocal() throws Exception {
		if (lstm.getbPTT() == null) {
			lstm.setbPTT(lstm.createBPTT()); 
		}
		
		err = 0;
		for (int i = 0; i < mb; i++) {
			if (!is.hasNext()) {
				is.reset();
			}
			is.next(); 
			err = err + lstm.runEpich(is.getSampleTT(), is.getTarget());
//			cnn.getcNNBP().runTrainEpich(new IDataMatrix[] { dm }, dm.getTarget());
//			err = err + cnn.getcNNBP().getStdError();
			
		}
	}

	@Override
	public Error getError() {
		error.setErr(err);
		return error;
	}

	@Override
	public void setSubject(Object obj) {
		if (obj instanceof LSTM) {
			lstm = (LSTM) obj;
		} else {
			swWs = (double [][]) obj;
//			cnnMerger.merge(cnn, swWs, true);
			translator.update(lstm.getCfg(), swWs, true);
		}	
	}
	
	LSTMWwArrayTranslator translator = new LSTMWwArrayTranslator();
	double [][] wWs;
	double [][] swWs;
	@Override
	public Object getLocalSubject() {
		if (wWs == null) {			
		}
		wWs = new double[lstm.getCfg().getLayers().length][];
		translator.update(lstm.getCfg(), wWs, false);
		return wWs;
	}

}
