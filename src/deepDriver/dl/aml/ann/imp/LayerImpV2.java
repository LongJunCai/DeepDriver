package deepDriver.dl.aml.ann.imp;

import java.io.Serializable;


import deepDriver.dl.aml.ann.ANNCfg;
import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.ann.ILayer;
import deepDriver.dl.aml.ann.INeuroUnit;
import deepDriver.dl.aml.ann.InputParameters;
import deepDriver.dl.aml.cnn.FlatAcf;
import deepDriver.dl.aml.costFunction.ICostFunction;
import deepDriver.dl.aml.distribution.modelParallel.PartialCallback;
import deepDriver.dl.aml.distribution.modelParallel.ThreadParallel;

public class LayerImpV2 extends LayerImp implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	ANNCfg aNNCfg;	

	public ANNCfg getaNNCfg() {
		return aNNCfg;
	}

	public void setaNNCfg(ANNCfg aNNCfg) {
		this.aNNCfg = aNNCfg;
	}
	
	@Override
	public void updateValues4FirstLayer(double[][] input) {
		updateValues4PartialFirstLayer(input, 0, -1);
	}

	public void updateValues4PartialFirstLayer(double[][] input, int offset, int length) {
		int cnt = 0;
		if (length <= 0) {
			length = input[0].length;
		}
		for (int i = 0; i < input.length; i++) {
			for (int j = offset; j < offset + length; j++) {
				if (j > getNeuros().size() -1) {
					NeuroUnitImp vONeuro = createNeuroUnitImp();
					this.getNeuros().add(vONeuro);
					vONeuro.setAas(new double[input.length]);
					vONeuro.setZzs(new double[input.length]);
					vONeuro.setActivationFunction(new FlatAcf());
					vONeuro.buildup(input, j);
				} 
				NeuroUnitImp vONeuro = (NeuroUnitImp) getNeuros().get(j);
				if (vONeuro.getAas().length != input.length) {
					vONeuro.setAas(new double[input.length]);
					vONeuro.setZzs(new double[input.length]);
				}				
//				vONeuro.setResult(i, input[i][j]);
				vONeuro.getAas()[i] = input[i][j];
			}
		}
	}
	
	double [] rs;
	
	public double[] getRs() {
		return rs;
	}

	public void setRs(double[] rs) {
		this.rs = rs;
	}
	
	ICostFunction costFunction;	
	
	public ICostFunction getCostFunction() {
		return costFunction;
	}
	
	public void setCostFunction(ICostFunction costFunction) {
		costFunction.setLayer(this);
		this.costFunction = costFunction;
	}
	
	double dropOut = 0;
	boolean firstBp = true;
	
	public void backPropagation(double [][] finalResult, InputParameters parameters) {
		if (enableDropOut()) {
			if (firstBp) {
				backPropagation4PartialLayer(finalResult, parameters);
				firstBp = false;
			}
			bp4DropOut(finalResult, parameters);
		} else {
			backPropagation4PartialLayer(finalResult, parameters);
//			for (int i = 0; i < getNeuros().size(); i++) {
//				INeuroUnit neuro = getNeuros().get(i);
//				neuro.backPropagation(getPreviousLayer() == null? null :this.getPreviousLayer().getNeuros(), getNextLayer() == null? null : this.getNextLayer().getNeuros(), finalResult, parameters);
//			}
		}		
		if (getNextLayer() == null && costFunction != null) {
			for (int i = 0; i < finalResult.length; i++) {
				costFunction.setzZIndex(i);
				costFunction.setTarget(finalResult[i]);
				costFunction.caculateCostError();
			}
		}
	}
	
	public void backPropagation4PartialLayer(final double [][] finalResult, final InputParameters parameters) {
		int tn = 1;
		if (aNNCfg == null || (tn = aNNCfg.getThreadsNum()) <= 1) {
			backPropagation4PartialLayer(finalResult, parameters,
				0, neuros.size());
		} else {
			threadParallel.runMutipleThreads(neuros.size(), new PartialCallback() {
				public void runPartial(int offset, int runLen) {
					backPropagation4PartialLayer(finalResult, parameters, offset,
							runLen);
				}
			}, tn);
		}
	}
	
	public void backPropagation4PartialLayer(double [][] finalResult, InputParameters parameters,
			int offset, int length) {
		for (int i = offset; i < offset + length; i++) {
			INeuroUnit neuro = getNeuros().get(i);
			neuro.backPropagation(getPreviousLayer() == null? null :this.getPreviousLayer().getNeuros(), getNextLayer() == null? null : this.getNextLayer().getNeuros(), finalResult, parameters);
		}
	}
	
	private void bp4DropOut(final double [][] finalResult, final InputParameters parameters) {
		int tn = 1;
		if (aNNCfg == null || (tn = aNNCfg.getThreadsNum()) <= 1) {
			bp4DropOut4Partial(finalResult, parameters,
				0, neuros.size());
		} else {
			threadParallel.runMutipleThreads(neuros.size(), new PartialCallback() {
				public void runPartial(int offset, int runLen) {
					bp4DropOut4Partial(finalResult, parameters, offset,
							runLen);
				}
			}, tn);
		}
	}
	
	private void bp4DropOut4Partial(double [][] finalResult, InputParameters parameters, 
			int offset, int length) {
		for (int i = offset; i < offset + length; i++) {
			NeuroUnitImpV3 neuro = (NeuroUnitImpV3) neuros.get(i);
			if (enableDropOut()) {
				if (aNNCfg.isTesting()) {//by default, there is no bp in testing.	
				} else {
					if (neuro.isDropOut()) {
						double [] dzs = neuro.getDeltaZ();
						for (int j = 0; j < dzs.length; j++) {
							dzs[j] = 0;
						}
						// no further bp
					} else {
						neuro.backPropagation(getPreviousLayer() == null? null :this.getPreviousLayer().getNeuros(), getNextLayer() == null? null : this.getNextLayer().getNeuros(), finalResult, parameters);
					}
				}
			}
		}
	}
	
//	int zZIndex = 0;
	
	boolean first = true;
	
	public void forwardPropagation4PartialLayer(double [][] input, 
			int offset, int length) {
		if (getPreviousLayer() == null) {
			updateValues4PartialFirstLayer(input, offset, length);
			return;
		}
		for (int i = offset; i < offset + length; i++) {
			INeuroUnit neuro = neuros.get(i);
			neuro.forwardPropagation(this.getPreviousLayer().getNeuros(), input);
		}
	}
	
	ThreadParallel threadParallel = new ThreadParallel();
	
	public void forwardPropagation4PartialLayer(final double [][] input) {
		int tn = 1;
		if (aNNCfg == null || (tn = aNNCfg.getThreadsNum()) <= 1) {
			forwardPropagation4PartialLayer(input,
				0, neuros.size());
		} else {
			threadParallel.runMutipleThreads(neuros.size(), new PartialCallback() {
				public void runPartial(int offset, int runLen) {
					forwardPropagation4PartialLayer(input, offset,
							runLen);
				}
			}, tn);
		}
	}
	
	double [][] rss;
	@Override
	public void forwardPropagation(double[][] input) {
		if (enableDropOut()) {
			if (first) {
//				super.forwardPropagation(input);
				forwardPropagation4PartialLayer(input);
				first = false;
			}
			fwd4DropOut(input);
		} else {
//			super.forwardPropagation(input);
			forwardPropagation4PartialLayer(input);
		}		
		if (getNextLayer() == null && costFunction != null) {
			rss = new double[input.length][];
			for (int i = 0; i < input.length; i++) {				
				costFunction.setzZIndex(i);
				rss[i] = costFunction.activate();
				rs = rss[i];
			}
		}
	}
		
	public double[][] getRss() {
		return rss;
	}

	public void setRss(double[][] rss) {
		this.rss = rss;
	}

	private boolean enableDropOut() {
		if (aNNCfg != null && aNNCfg.getDropOut() > 0
				&& getNextLayer() != null && getPreviousLayer() != null) {
			return true;
		}
		return false;
	}
	
	boolean typicalDropout = false;
	private void fwd4DropOut(final double[][] input) {
		int tn = 1;
		if (aNNCfg == null || (tn = aNNCfg.getThreadsNum()) <= 1) {
			fwd4DropOut4PartialLayer(input,
				0, neuros.size());
		} else {
			threadParallel.runMutipleThreads(neuros.size(), new PartialCallback() {
				public void runPartial(int offset, int runLen) {
					fwd4DropOut4PartialLayer(input, offset,
							runLen);
				}
			}, tn);
		}
	}
	private void fwd4DropOut4PartialLayer(double[][] input, int offset, int length) {
		// dorp out, only for hidden layer
		for (int i = offset; i < offset + length; i++) {
			NeuroUnitImpV3 neuro = (NeuroUnitImpV3) neuros.get(i);
			if (enableDropOut()) {
				if (aNNCfg.isTesting()) {
					neuro.forwardPropagation(this.getPreviousLayer().getNeuros(), input);
					if (!typicalDropout) {
						double [] aAs = neuro.getAas();
						for (int j = 0; j < aAs.length; j++) {
							aAs[j] = aAs[j] * (1.0 - aNNCfg.getDropOut());
						}						
					}					
				} else {
					double a = NeuroUnitImp.random.nextDouble();
					if (a < aNNCfg.getDropOut()) {
						double [] aAs = neuro.getAas();
						for (int j = 0; j < aAs.length; j++) {
							aAs[j] = 0;
						}
//						neuro.getAas()[zZIndex] = 0;// no further output
						neuro.setDropOut(true);						
					} else {
						if (getPreviousLayer() != null) {
							neuro.forwardPropagation(this.getPreviousLayer().getNeuros(), input);
							if (typicalDropout) {
								double [] aAs = neuro.getAas();
								for (int j = 0; j < aAs.length; j++) {
									aAs[j] = aAs[j]
											/ (1.0 - aNNCfg.getDropOut());
								}
//								neuro.getAas()[zZIndex] = neuro.getAas()[zZIndex]
//									/ (1.0 - aNNCfg.getDropOut());
							}
						}						
						neuro.setDropOut(false); 
					}
				}
			}			
		}
	}
	
	@Override
	public double getStdError(double[][] result) {
		if (getNextLayer() == null && costFunction != null) {
			double err = 0;
			for (int i = 0; i < result.length; i++) {
				costFunction.setzZIndex(i);
				costFunction.setTarget(result[i]);
				err = err + costFunction.caculateStdError(); 
			}
//			costFunction.setTarget(result[0]);
//			return costFunction.caculateStdError(); 
			return err;
		}
		return super.getStdError(result);
	}

	public NeuroUnitImp createNeuroUnitImp() {
		return new NeuroUnitImpV3(this);
	}		

	@Override
	public void buildup(ILayer previousLayer, double[][] input, IActivationFunction acf, boolean isLastLayer, int neuroCount) {
		setPreviousLayer(previousLayer);
		if (previousLayer != null) {
			previousLayer.setNextLayer(this);
		}
		if (previousLayer == null) {
			updateValues4FirstLayer(input);
		} else {					//input[0].length	
			for (int i = 0; i < neuroCount; i++) {
				NeuroUnitImp neuroUnitImp = createNeuroUnitImp();
				neuroUnitImp.buildup(input, i);
				neuroUnitImp.setActivationFunction(acf);
				getNeuros().add(neuroUnitImp); 
			}			
		}
	}

}
