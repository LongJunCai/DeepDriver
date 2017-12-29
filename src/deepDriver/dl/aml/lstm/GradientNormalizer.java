package deepDriver.dl.aml.lstm;

import java.io.Serializable;

public class GradientNormalizer implements IRNNLayerVisitor, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	int CaculateNorm = 1;
	int NormGradient = 2;
	int currentState = CaculateNorm;
	double norm = 0;
	public void checkGradient(LSTMConfigurator cfg) {
		currentState = CaculateNorm;
		norm = 0;
		IRNNLayer [] layers = cfg.getLayers();
		for (int i = 0; i < layers.length; i++) {
			layers[i].updateWw(this);
		}
		double b = norm;
		norm = Math.pow(norm, 0.5);
//		System.out.println("The norm is "+b+", now it is: "+norm);		
	}
	
	double threshold = 5;
	
	public void normGradient(LSTMConfigurator cfg, double threshold) {
		this.threshold = threshold;
		checkGradient(cfg);
		if (norm < threshold ) {
			return;
		} else {
			System.out.println("The norm is: "+norm+", will norm it.");
		}
		currentState = NormGradient;
		IRNNLayer [] layers = cfg.getLayers();
		for (int i = 0; i < layers.length; i++) {
			layers[i].updateWw(this);
		}
		checkGradient(cfg);	
	}

	@Override
	public void updateWw4RNNLayer(RNNLayer layer) {
		RNNNeuroVo [] rnnVos = layer.getRNNNeuroVos();
		for (int i = 0; i < rnnVos.length; i++) {
			if (currentState == CaculateNorm) {
				norm = norm + caculateNorm(rnnVos[i]);
			} else if (currentState == NormGradient) {
				normWws(rnnVos[i]);
			}			
		}
	}

	@Override
	public void updateWw4RNNLayer(LSTMLayer layer) {
		IBlock [] blocks = layer.getBlocks();
		for (int i = 0; i < blocks.length; i++) {
			IBlock block = blocks[i];				
			ICell [] cells = block.getCells();
			IInputGate igate = block.getInputGate();
			IOutputGate ogate = block.getOutPutGate();
			IForgetGate fgate = block.getForgetGate();
			for (int j = 0; j < cells.length; j++) {
				ICell cell = cells[j];	
				if (currentState == CaculateNorm) {
					norm = norm + caculateNorm(cell); 
				} else if (currentState == NormGradient) {
					normWws(cell); 
				}				
			}
			if (currentState == CaculateNorm) {
				norm = norm + caculateNorm(igate); 			
				norm = norm + caculateNorm(ogate);			
				norm = norm + caculateNorm(fgate); 
			} else if (currentState == NormGradient) {
				normWws(igate); 			
				normWws(ogate);			
				normWws(fgate); 
			}
			 			
		}	
	}
	
	public void normWws(IRNNNeuroVo rnnVo) {
		normWws(rnnVo.getDeltaWWs());
		normWws(rnnVo.getDeltaLwWs());
		normWws(rnnVo.getDeltaRwWs());
	}
	
	public void normWws(double [] wWs) {
		if (wWs != null) {
			for (int i = 0; i < wWs.length; i++) {
				wWs[i] = wWs[i]/norm * threshold;
			}
		}
	}
	
	public double caculateNorm(IRNNNeuroVo rnnVo) {
		double norm = 0;
		norm = norm + normOfWws(rnnVo.getDeltaWWs());
		norm = norm + normOfWws(rnnVo.getDeltaLwWs());
		norm = norm + normOfWws(rnnVo.getDeltaRwWs());		
		return norm;		
	}
	
	public double normOfWws(double [] wWs) {
		double norm = 0;
		if (wWs != null) {
			for (int i = 0; i < wWs.length; i++) {
				norm = wWs[i] * wWs[i];
			}
		}
		return norm;
	}

	@Override
	public void updateWw4RNNLayer(ProjectionLayer layer) {
		// TODO Auto-generated method stub
		
	}

}
