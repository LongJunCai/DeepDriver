package deepDriver.dl.aml.cnn.distribution;

import deepDriver.dl.aml.cnn.CNNParaMerger;
import deepDriver.dl.aml.cnn.CacheAbleDataStream;
import deepDriver.dl.aml.cnn.ConvolutionNeuroNetwork;
import deepDriver.dl.aml.cnn.IDataMatrix;
import deepDriver.dl.aml.cnn.IDataStream;
import deepDriver.dl.aml.common.distribution.Linkable;
import deepDriver.dl.aml.common.distribution.LinkableDataStream;
import deepDriver.dl.aml.distribution.Error;
import deepDriver.dl.aml.distribution.Slave;

public class CNNSlave extends Slave {
	
	ConvolutionNeuroNetwork cnn;
	static int mb = 512;
	IDataStream is;
	@Override
	public void setTask(Object obj) throws Exception {
		if (obj instanceof Linkable) {
			Linkable root = (Linkable) obj;	
			is = new LinkableDataStream((CacheAbleDataStream) root);
		} else if (obj instanceof IDataStream) {
			is = (IDataStream) obj;			
		} 		
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
			if (!is.hasNext()) {
				is.reset();
			}
			IDataMatrix [] dm = is.next();
			if (dm == null) {
				continue;
			}
			if (dm.length <= ConvolutionNeuroNetwork.MatrixTargetIndex
					|| dm[ConvolutionNeuroNetwork.MatrixTargetIndex] == null) {
				System.out.println(i+" is null");
				continue;
			} 
			double [] ta = dm[ConvolutionNeuroNetwork.MatrixTargetIndex].getTarget();
			cnn.getcNNBP().runTrainEpich(dm, ta);
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
