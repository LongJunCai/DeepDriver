package deepDriver.dl.aml.contrib.MNIST;

import java.util.Random;

import deepDriver.dl.aml.cnn.DataMatrix;
import deepDriver.dl.aml.cnn.IDataMatrix;
import deepDriver.dl.aml.cnn.IDataStream;

public class MnistDataStream implements IDataStream {

	int type; // 0 for training, 1 for testing;
	int size; // Number of Training Images
	int cnt;  // Image Count
	int ml = 28;  // 28* 28
	int nLabel = 10; // number of digit Label
	int[][] imgTrain;
	int[][] imgTest;
	int[] lbTrain;
	int[] lbTest;
	double omax = 256.0;
	
	// type 0: training set, type 1 testing set;
	public MnistDataStream(MnistLoader loader, int type) {
		super();
		this.imgTrain = loader.getImgTrain();
		this.imgTest = loader.getImgTest();
		this.lbTrain = loader.getLabelTrain();
		this.lbTest = loader.getLabelTest();
		this.type = type;
		this.size = (type == 0)?imgTrain.length:imgTest.length;
	}

	Random  rd = new Random(System.currentTimeMillis());
	@Override
	public IDataMatrix next() {
		cnt++;
		double l = (double) size;
		int ri  = (int) (rd.nextDouble() * l);
		if (ri == l) {//exclusive, so no need to worry about it.
			ri = ri - 1;
		}
		return getIDataMatrix(ri,type);
	}

	@Override
	public IDataMatrix next(Object pos) {
		cnt++;
		int id  = (Integer) pos;
		return getIDataMatrix(id, type);
	}

	@Override
	public boolean hasNext() {
		if (cnt <= size - 1) {
			return true;
		}
		return false;
	}

	@Override
	public boolean reset() {
		cnt = 0;
		return true;
	}

	private double[] LabelToVector(int label, int nLable) {
		// label 0-9
		double[] target = new double[nLable]; 
		target[label] = 1.0;
		return target;
	}
	
	public IDataMatrix getIDataMatrix(int id, int type) {
		// id is index of training image
		int[] pixel; // pixel of id-th image
		int label;
		if (type == 0) {
			pixel = imgTrain[id];
			label = lbTrain[id];
		} else {
			pixel = imgTest[id];
			label = lbTest[id];
		}
		DataMatrix dataMatrix = new DataMatrix();
		// Label
		double[] ta = LabelToVector(label, nLabel);
		dataMatrix.setTarget(ta);
		// Image
		double [][] m = new double[ml][ml];
		for (int i = 0; i < ml; i++) {
			for (int j = 0; j < ml; j++) {
				m[i][j] = ((double) pixel[i * ml+j])/omax;
			}	
		}
		dataMatrix.setMatrix(m);
		return dataMatrix;
	}
	
}
