package deepDriver.dl.aml.cnn.distribution;

import deepDriver.dl.aml.cnn.CNNParaMerger;
import deepDriver.dl.aml.cnn.ConvolutionNeuroNetwork;
import deepDriver.dl.aml.cnn.IDataMatrix;
import deepDriver.dl.aml.cnn.IDataStream;
import deepDriver.dl.aml.distribution.Error;
import deepDriver.dl.aml.distribution.Slave;

public class CNNSlave extends Slave {
	
	ConvolutionNeuroNetwork cnn;
	static int mb = 1024;
	IDataStream is;
	@Override
	public void setTask(Object obj) throws Exception {
		is = (IDataStream) obj;
	}

	double err = 0;
	Error error = new Error();
	@Override
	public void trainLocal() throws Exception {
		if (cnn.getcNNBP() == null) {
			cnn.setcNNBP(cnn.createCNNBP()); 
		}
		
		err = 0;
		for (int i = 0; i < mb; i++) {
			if (is.hasNext()) {
				is.reset();
			}
			IDataMatrix dm = is.next();
			if (dm == null) {
				continue;
			}
			cnn.getcNNBP().runTrainEpich(new IDataMatrix[] { dm }, dm.getTarget());
			err = err + cnn.getcNNBP().getStdError();
			
		}
	}

	@Override
	public Error getError() {
		error.setErr(err);
		return error;
	}

	@Override
	public void setSubject(Object obj) {
		if (obj instanceof ConvolutionNeuroNetwork) {
			cnn = (ConvolutionNeuroNetwork) obj;
		} else {
			swWs = (double [][]) obj;
			cnnMerger.merge(cnn, swWs, true);
		}	
	}
	
	CNNParaMerger cnnMerger = new CNNParaMerger();
	double [][] wWs;
	double [][] swWs;
	@Override
	public Object getLocalSubject() {
		if (wWs == null) {			
		}
		wWs = new double[cnn.getCfg().getLayers().length][];
		cnnMerger.merge(cnn, wWs, false);
		return wWs;
	}

}
