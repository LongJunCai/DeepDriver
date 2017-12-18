package deepDriver.dl.aml.cnn.img;

import java.util.Random;

import deepDriver.dl.aml.cnn.DataMatrix;
import deepDriver.dl.aml.cnn.IDataMatrix;
import deepDriver.dl.aml.cnn.IDataStream;

public class W2VDataStream implements IDataStream {
	
	CsvImgLoader imgLoader;
	int cnt;
	
	String label = "label";
	String splitter = ",";
	
	double omax = 20;
	//double omin = 0;
	//double min = -0.1;
	//double max = 1.175;
	
	boolean binaryNormalize = false;
	
	Random  rd = new Random(System.currentTimeMillis());
	
	@Override
	public IDataMatrix next() {
		cnt++;
		double l = imgLoader.getImgs().size();
		int ri  = (int) (rd.nextDouble() * l);
		if (ri == l) {//exclusive, so no need to worry about it.
			ri = ri - 1;
		}
		String s = imgLoader.getImgs().get(ri);
		boolean hasHeader = false;
		if (imgLoader.header != null) {
		    hasHeader = imgLoader.header.startsWith(label);
        }
		return constructIDataMatrix(s, hasHeader);
	}
	
	public IDataMatrix next(Object pos) {
		cnt++;
		int ri = (Integer) pos;
		String s = imgLoader.get(ri);
		return constructIDataMatrix(s, imgLoader.header.startsWith(label));
	}
	
	int rLength;
	
	int wordsNum = 3;
	
	int fixedRow = -1;	
	
	public int getFixedRow() {
        return fixedRow;
    }

    public void setFixedRow(int fixedRow) {
        this.fixedRow = fixedRow;
    }

    public IDataMatrix constructIDataMatrix(String s, boolean h) {
		DataMatrix dataMatrix = new DataMatrix();
		String [] arr = s.split(splitter);
		double [] ta = new double[tLength];
		int cnt = 0;
		int ml = 0;
		if (h) {
			int t = Integer.parseInt(arr[cnt ++]);
			ta[t - 1] = 1;
//			ml = (int) Math.sqrt(arr.length - 1);
			dataMatrix.setTarget(ta);
		} else {
//			ml = (int) Math.sqrt(arr.length);
			dataMatrix.setTarget(null);
		}
		if ((arr.length - 1)/rLength < wordsNum) {
            return null;
        }
		int r = (arr.length - 1)/rLength;
		if (fixedRow > 0 ) {
            r = fixedRow;
        }
//		r = 20;
		double [][] m = new double[r][rLength];
		for (int i = 0; i < m.length; i++) {
			m[i] = new double[rLength];
			for (int j = 0; j < m[i].length; j++) {			    
				double a = 0;
				if (cnt < arr.length) {
                    a = Double.parseDouble(arr[cnt ++]);
                }				
//				a = a/omax;
//				sum = sum + a/omax;
//				if (binaryNormalize) {
//					if (a > 0) {
//						m[i][j] = max;
//					} else {
//						m[i][j] = min;
//					}
//				} else {
//					m[i][j] = (a - omin)/(omax - omin) *
//						(max - min) + min;
//				}
				m[i][j] = a;//no normalizing here.
			}
		}
		dataMatrix.setMatrix(m);
		return dataMatrix;
	}
	
	public boolean reset() {
		cnt = 0;
		return true;
	}
	
	int tLength;
	
	public W2VDataStream(CsvImgLoader imgLoader, int tLength, int rLength) {
		super();
		this.imgLoader = imgLoader;
		this.tLength = tLength;
		this.rLength = rLength;
	}

	@Override
	public boolean hasNext() {
		if (cnt <= imgLoader.size() - 1) {
			return true;
		}
		return false;
	}
	
	public static void main(String[] args) throws Exception {
		String file = "E://models//cnn//sentiment3.0//cnn-sentiment-training-26000.txt";
		CsvImgLoader imgLoader = new CsvImgLoader();
		imgLoader.loadImg(file);
		int kLength = 3; //绫诲埆鏁�
		int cnt = 0;
		W2VDataStream w2vDataStream = new W2VDataStream(imgLoader, kLength, 200);
		while (w2vDataStream.hasNext()) {
			w2vDataStream.next();
			cnt ++;
			if (cnt % 4000 == 0) {
				System.out.println("Run "+cnt+" already.");
			}
		}
		System.out.println("sum: "+", "+ cnt );
	}
	
	
	


}
