package deepDriver.dl.aml.cnn.img;

import deepDriver.dl.aml.cnn.IDataMatrix;

public class W2VDataStreamV24Test extends W2VDataStreamV2 {
    
    public W2VDataStreamV24Test(CsvImgLoader imgLoader, int tLength, int rLength) {
        super(imgLoader, tLength, rLength);
    }

    public IDataMatrix [] next() {
        return new IDataMatrix [] {getIDataMatrix(cnt++)};
    }
    

}
