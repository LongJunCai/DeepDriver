package deepDriver.dl.aml.cnn.img;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import deepDriver.dl.aml.cnn.DataMatrix;
import deepDriver.dl.aml.cnn.IDataMatrix;
import deepDriver.dl.aml.cnn.IDataStream;

public class ImgDataStream implements IDataStream {
	
	CsvImgLoader imgLoader;
	int cnt;
	
	String label = "label";
	String splitter = ",";
	
	double omax = 256;
	double omin = 0;
	double min = -0.1;
	double max = 1.175;
	
	boolean binaryNormalize = false;
	
	boolean preLoad = false;
	List<IDataMatrix> dmCache = new ArrayList<IDataMatrix>();
	
	public void preLoad() {
    	preLoad = true;
    	for (int i = 0; i < imgLoader.size(); i++) {
    		String s = imgLoader.get(i);
    		IDataMatrix dm = constructIDataMatrix(s, imgLoader.header.startsWith(label));
    		dmCache.add(dm);
		}
	}
	
	public IDataMatrix getIDataMatrix(int id) {
		if (preLoad && dmCache.size() > 0) {
			return dmCache.get(id);
		} else {
//			String s = imgLoader.imgs.get(id);
			String s = imgLoader.get(id);
			return constructIDataMatrix(s, imgLoader.header.startsWith(label));
		}
	}
	
	Random  rd = new Random(System.currentTimeMillis());
	@Override
	public IDataMatrix [] next() {
		cnt++;
//		double l = imgLoader.imgs.size();
		double l = imgLoader.size();
		int ri  = (int) (rd.nextDouble() * l);
		if (ri == l) {//exclusive, so no need to worry about it.
			ri = ri - 1;
		}
//		String s = imgLoader.imgs.get(ri);
//		return constructIDataMatrix(s, imgLoader.header.startsWith(label));
		return new IDataMatrix [] {getIDataMatrix(ri)};
	}
	
	public IDataMatrix [] next(Object pos) {
		cnt++;
		int ri  = (Integer) pos;
		return new IDataMatrix [] {getIDataMatrix(ri)};
	}
	
	double sum;
	
	public IDataMatrix constructIDataMatrix(String s, boolean h) {
		DataMatrix dataMatrix = new DataMatrix();
		String [] arr = s.split(splitter);
		double [] ta = new double[tLength];
		int cnt = 0;
		int ml = 0;
		if (h) {
			int t = Integer.parseInt(arr[cnt ++]);
			ta[t] = 1;
			ml = (int) Math.sqrt(arr.length - 1);
			dataMatrix.setTarget(ta);
		} else {
			ml = (int) Math.sqrt(arr.length);
			dataMatrix.setTarget(null);
		}
		
		double [][] m = new double[ml][ml];
		for (int i = 0; i < m.length; i++) {
			m[i] = new double[ml];
			for (int j = 0; j < m[i].length; j++) {
				double a = Double.parseDouble(arr[cnt ++]);
				sum = sum + a/omax;
				if (binaryNormalize) {
					if (a > 0) {
						m[i][j] = max;
					} else {
						m[i][j] = min;
					}
				} else {
					m[i][j] = (a - omin)/(omax - omin) *
						(max - min) + min;
				}
//				m[i][j] = a;//no normalizing here.
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
	
	public ImgDataStream(CsvImgLoader imgLoader, int tLength) {
		super();
		this.imgLoader = imgLoader;
		this.tLength = tLength;
	}

	@Override
	public boolean hasNext() {
//		if (cnt <= imgLoader.imgs.size() - 1) {
		if (cnt <= imgLoader.size() - 1) {
			return true;
		}
		return false;
	}
	
	public static void main(String[] args) throws Exception {
		String file = "D:\\6.workspace\\ANN\\cnn\\kaggleTest\\modelTrain";
		CsvImgLoader imgLoader = new CsvImgLoader();
		imgLoader.loadImg(file);
		int kLength = 10;
		int cnt = 0;
		ImgDataStream imgDataStream = new ImgDataStream(imgLoader, kLength);
		while (imgDataStream.hasNext()) {
			imgDataStream.next();
			cnt ++;
			if (cnt % 4000 == 0) {
				System.out.println("Run "+cnt+" already.");
			}
		}
		System.out.println("sum: "+imgDataStream.sum+", "+cnt * 28 * 28);
	}

	@Override
	public IDataStream[] splitStream(int segments) { 
		return null;
	}

	@Override
	public int splitCnt(int segments) {
		// TODO Auto-generated method stub
		return 0;
	}
	
	
	


}
