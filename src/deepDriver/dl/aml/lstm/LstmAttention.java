package deepDriver.dl.aml.lstm;

import deepDriver.dl.aml.attention.SoftAttention;
import deepDriver.dl.aml.cnn.ActivationFactory;
import deepDriver.dl.aml.lstm.imp.Cell;

public class LstmAttention {
	
	SoftAttention attention;
	public LstmAttention(int waSize, int uaSize, int maxLength) {
		attention = new SoftAttention(waSize, uaSize,  ActivationFactory.getAf().getTanh(), maxLength);
		cs = new double[maxLength][];
	}
	
	double [][] hj;
	public void passHj(LSTMLayer layer, int tLength) {
		hj = new double[tLength][];
		if (layer instanceof BiLstmLayer) {
			BiLstmLayer biLayer = (BiLstmLayer) layer;
			biLayer.reverseOpposite(tLength);
		}
		for (int i = 0; i < tLength; i++) {
			hj[i] = new double[layer.getCells().length]; 
			for (int j = 0; j < hj[i].length; j++) {
				hj[i][j] = layer.getCells()[j].getNvTT()[i].getaA();
			}			
		}
		if (layer instanceof BiLstmLayer) {
			BiLstmLayer biLayer = (BiLstmLayer) layer;
			biLayer.reverseBackOpposite();
		}
	}
	
	
	double [][] cs;
	public void fTT4RNNLayerAttention(LSTMLayer layer, int t) {
		LayerCfg lc = layer.getLc();
		if (lc == null || lc.getAttentionLength() <= 0) {
			return ;
		} 
		double [] st_1 = new double[layer.getCells().length]; 
		if (t > 0) {
			for (int i = 0; i < st_1.length; i++) { 	
				st_1[i] = layer.getCells()[i].getSc()[t - 1];
			}
		}
		
		cs[t] = attention.forward(st_1, hj, t); 
	}
	
	public double fttAttentionSc(LSTMLayer layer, RNNNeuroVo rnVo, int t) {
		LayerCfg lc = layer.getLc(); 
		if (lc == null || lc.getAttentionLength() <= 0 || t == 0) {
			return 0;
		} 
		double [] wWs = rnVo.getwWas();
		return attention.dot(wWs, cs[t]);
	}
	
	public void bp4RNNLayerAttention(LSTMLayer layer, int t) {
		LayerCfg lc = layer.getLc();
		if (lc == null || lc.getAttentionLength() <= 0) {
			return ;
		} 
		double [] st_1 = new double[layer.getCells().length]; 
		for (int i = 0; i < st_1.length; i++) { 	
			st_1[i] = layer.getCells()[i].getSc()[t - 1];
		}
		double [] c = cs[t];
		double [] dc = new double[cs[t].length];
		ICell [] cells = layer.getCells();
		for (int i = 0; i < cells.length; i++) {
			Cell cell = (Cell) cells[i];
			for (int j = 0; j < dc.length; j++) {
				dc[j] = dc[j] + cell.getDeltaSc()[t] * cell.getwWas()[j];
				if (j == 0) {
					cell.getDeltaWwas()[j] = cell.getDeltaSc()[t] * c[j];
				} else {
					cell.getDeltaWwas()[j] = cell.getDeltaWwas()[j] + cell.getDeltaSc()[t] * c[j];
				}
			}
		}
		attention.bp(st_1, dc, t);
	}
	
	public double getDeltaSct_1(LSTMLayer layer, int pos) {
		LayerCfg lc = layer.getLc();
		if (lc == null || lc.getAttentionLength() <= 0) {
			return 0;
		} 
		return attention.getDsc()[pos];
	}
	
	public double[][] getDeltaHj() {
		return attention.getDhj();
	}
	
	double l;
	double m; 
	
	public double getL() {
		return l;
	}

	public void setL(double l) {
		this.l = l;
	}

	public double getM() {
		return m;
	}

	public void setM(double m) {
		this.m = m;
	}
	
	public void updateWw(LSTMLayer layer) {
		LayerCfg lc = layer.getLc();
		if (lc == null || lc.getAttentionLength() <= 0) {
			return;
		} 
		ICell [] cells = layer.getCells();
		for (int i = 0; i < cells.length; i++) {
			Cell cell = (Cell) cells[i];
			for (int j = 0; j < cell.getwWas().length; j++) {
				cell.getwWas()[j] = cell.getwWas()[j] - l * cell.getDeltaWwas()[j];
			}
		}
		attention.updateWw();
	}

}
