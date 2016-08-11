package deepDriver.dl.aml.lstm.data;

import deepDriver.dl.aml.lstm.IBlock;
import deepDriver.dl.aml.lstm.ICell;
import deepDriver.dl.aml.lstm.IRNNLayer;
import deepDriver.dl.aml.lstm.IRNNLayerVisitor;
import deepDriver.dl.aml.lstm.IRNNNeuroVo;
import deepDriver.dl.aml.lstm.LSTMLayer;
import deepDriver.dl.aml.lstm.ProjectionLayer;
import deepDriver.dl.aml.lstm.RNNLayer;

public class CfgDataTransfer implements IRNNLayerVisitor {
	
	static int CaculateNum = 1;
	static int GenerateData = 2;
	static int CopyData2Cfg = 3;
	
	static int MergeDeltaWw2Cfg = 4;
	
	int cState = CaculateNum;
	
	int neuroNum;
	double [][][] cData = null;
	int cDataIndex = 0;
	public LSTMCfgData loadCfg(IRNNLayer [] layer) {
		neuroNum = 0;
		cDataIndex = 0;
		cData = null;
		LSTMCfgData cfgData = new LSTMCfgData();
		for (int i = 1; i < layer.length; i++) {
			cState = CaculateNum;
			layer[i].updateWw(this);
		}
		cData = new double[neuroNum][][];
		for (int i = 1; i < layer.length; i++) {
			cState = GenerateData;
			layer[i].updateWw(this);
		}
		cfgData.setCfg(cData);
		return cfgData;		
	}
	
	public void copyData2Cfg(LSTMCfgData cData, IRNNLayer [] layer) {
		cDataIndex = 0;
		this.cData = cData.getCfg();
		for (int i = 1; i < layer.length; i++) {
			cState = CopyData2Cfg;
			layer[i].updateWw(this);
		}		
	}
	
	LSTMCfgData [] cDatas;
	public void mergeDeltaWw2Cfg(LSTMCfgData [] cDatas, IRNNLayer [] layer) {
		cDataIndex = 0;
		this.cDatas = cDatas;
		for (int i = 1; i < layer.length; i++) {
			cState = MergeDeltaWw2Cfg;
			layer[i].updateWw(this);
		}		
	}

	@Override
	public void updateWw4RNNLayer(RNNLayer layer) {
		if (cState == CaculateNum) {
			neuroNum = neuroNum + layer.getRNNNeuroVos().length;
		} else if (cState == GenerateData) {
			IRNNNeuroVo [] vos = layer.getRNNNeuroVos();
			for (int i = 0; i < vos.length; i++) {
				copyFromIRNNNeuroVo(vos[i]);
			}
		} else if (cState == CopyData2Cfg) {
			IRNNNeuroVo [] vos = layer.getRNNNeuroVos();
			for (int i = 0; i < vos.length; i++) {
				copy2IRNNNeuroVo(vos[i]);
			}			
		} else if (cState == MergeDeltaWw2Cfg) {
			IRNNNeuroVo [] vos = layer.getRNNNeuroVos();
			for (int i = 0; i < vos.length; i++) {
				mergeData2IRNNNeuroVo(vos[i]);
			}	
		}
		
	}


	@Override
	public void updateWw4RNNLayer(LSTMLayer layer) {		
		IBlock [] blocks = layer.getBlocks();
		for (int i = 0; i < blocks.length; i++) {
			IBlock block = blocks[i];
			if (cState == CaculateNum) {
				neuroNum = neuroNum + 
						block.getCells().length + 3;
			} else if (cState == GenerateData) {
				copyFromIRNNNeuroVo(block.getInputGate());
				copyFromIRNNNeuroVo(block.getOutPutGate());
				copyFromIRNNNeuroVo(block.getForgetGate());
				ICell [] cells = block.getCells();
				for (int j = 0; j < cells.length; j++) {
					copyFromIRNNNeuroVo(cells[i]);
				}
			} else if (cState == CopyData2Cfg) {
				copy2IRNNNeuroVo(block.getInputGate());
				copy2IRNNNeuroVo(block.getOutPutGate());
				copy2IRNNNeuroVo(block.getForgetGate());
				ICell [] cells = block.getCells();
				for (int j = 0; j < cells.length; j++) {
					copy2IRNNNeuroVo(cells[i]);
				}
			}  else if (cState == MergeDeltaWw2Cfg) {
				mergeData2IRNNNeuroVo(block.getInputGate());
				mergeData2IRNNNeuroVo(block.getOutPutGate());
				mergeData2IRNNNeuroVo(block.getForgetGate());
				ICell [] cells = block.getCells();
				for (int j = 0; j < cells.length; j++) {
					mergeData2IRNNNeuroVo(cells[i]);
				}
			}
		}
	}
	

	private void mergeData2IRNNNeuroVo(IRNNNeuroVo t2rnvo) {
		for (int i = 0; i < cDatas.length; i++) {
			double [][] a = cDatas[i].getCfg()[cDataIndex];
			mergeWws(i == 0, a[0], t2rnvo.getwWs(), t2rnvo.getDeltaWWs(), cDatas.length);
			mergeWws(i == 0, a[1], t2rnvo.getLwWs(), t2rnvo.getDeltaLwWs(), cDatas.length);
			mergeWws(i == 0, a[2], t2rnvo.getRwWs(), t2rnvo.getDeltaRwWs(), cDatas.length);			
		}
		cDataIndex ++;
	}
	
	public void mergeWws(boolean isNew, double [] fWws, double [] t2Wws, double [] detalWws, double length) {
		if (fWws == null) {
			return ;
		}
		for (int i = 0; i < detalWws.length; i++) {
			if (isNew) {
				detalWws[i] = (t2Wws[i] - fWws[i])/length;
			} else {
				detalWws[i] = detalWws[i] + (t2Wws[i] - fWws[i])/length;
			}			
		}
	}
	
	public void copy2IRNNNeuroVo(IRNNNeuroVo t2rnvo) {		
		double [][] b = new double [][]{
				t2rnvo.getwWs(),t2rnvo.getLwWs(),t2rnvo.getRwWs()
				,t2rnvo.getDeltaWWs(),t2rnvo.getDeltaLwWs(), t2rnvo.getDeltaRwWs()
		};		
		for (int i = 0; i < b.length; i++) {
			 copy2Wws(cData[cDataIndex][i], b[i]);
		}		
		cDataIndex ++;
	}
	public void copy2Wws(double [] fData, double [] t2Wws) {
		if (fData == null) {
			return ;
		}
		for (int i = 0; i < t2Wws.length; i++) {
			t2Wws[i] = fData[i];
		}
	}
	
	public void copyFromIRNNNeuroVo(IRNNNeuroVo t2rnvo) {		
		double [][] b = new double [][]{
				t2rnvo.getwWs(),t2rnvo.getLwWs(),t2rnvo.getRwWs()
				,t2rnvo.getDeltaWWs(),t2rnvo.getDeltaLwWs(), t2rnvo.getDeltaRwWs()
		};		
		cData[cDataIndex] = new double[b.length][];
		for (int i = 0; i < b.length; i++) {
			cData[cDataIndex][i] = copyFromWws(b[i]);
		}		
		cDataIndex ++;
	}
	
	public double [] copyFromWws(double [] t2Wws) {
		if (t2Wws == null) {
			return null;
		}
		double [] a = new double[t2Wws.length];
		for (int i = 0; i < t2Wws.length; i++) {
			a[i] = t2Wws[i];
		}
		return a;
	}

	@Override
	public void updateWw4RNNLayer(ProjectionLayer layer) {
		// TODO Auto-generated method stub
		
	}

}
