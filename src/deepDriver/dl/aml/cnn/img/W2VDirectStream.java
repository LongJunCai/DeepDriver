package deepDriver.dl.aml.cnn.img;

import deepDriver.dl.aml.cnn.DataMatrix;
import deepDriver.dl.aml.cnn.IDataMatrix;
import deepDriver.dl.aml.cnn.IDataStream;

public class W2VDirectStream implements IDataStream {
    
     float [][] data;
     float scaler = 1;
    
    public W2VDirectStream(float [][] data, float scaler) {
        super();
        this.data = data;
        this.scaler = scaler;
    }
    
    int cnt = 0;
    double omax = 39.6717;
    double omin = -33.2884;
    double min = -1;
    double max = 1;

    public IDataMatrix next() {
        cnt ++;
        double [][] ndata = new double[data.length][];
        DataMatrix dataMatrix = new DataMatrix();
        dataMatrix.setMatrix(ndata);
        dataMatrix.setTarget(null);
        for (int i = 0; i < data.length; i++) {
            ndata[i] = new double[data[i].length];
            for (int j = 0; j < data[i].length; j++) {     
                ndata[i][j] = (data[i][j] * scaler - omin)/(omax - omin) *
                        (max - min) + min;
            }
        }
        return dataMatrix;
    }

    @Override
    public boolean hasNext() {
        return cnt < 1;
    }

    @Override
    public boolean reset() {
        cnt = 0;
        return true;
    }

	@Override
	public IDataMatrix next(Object pos) {
		return null;
	}

}
