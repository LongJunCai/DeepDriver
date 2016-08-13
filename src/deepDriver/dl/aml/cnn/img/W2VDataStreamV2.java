package deepDriver.dl.aml.cnn.img;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import deepDriver.dl.aml.cnn.DataMatrix;
import deepDriver.dl.aml.cnn.IDataMatrix;
import deepDriver.dl.aml.cnn.IDataStream;

public class W2VDataStreamV2 implements IDataStream {
	
	CsvImgLoader imgLoader;
	int cnt;
	
	String label = "label";
	String splitter = ",";
	
	double omax = 39.6717;
	double omin = -33.2884;
	double min = -1;
	double max = 1;
	static double [] oMax = {24.0946,25.0517,23.4528,23.6357,30.2098,28.0905,24.8686,24.0888,23.0161,29.5336,29.871,24.2471,24.0336,25.4088,22.7328,22.3861,26.6361,22.5261,24.1597,30.3767,29.9871,21.4303,25.1227,23.0713,24.3369,28.6088,25.6336,26.3686,27.0497,23.5839,28.9791,22.0371,22.3939,27.4695,21.9759,23.6064,22.7257,25.0283,24.0408,28.3288,23.7934,24.6584,26.8787,22.4681,25.5116,28.1734,24.8098,27.3996,31.6791,24.1734,24.0075,22.9568,27.489,26.2837,22.4962,23.8817,29.2783,30.2883,22.7744,24.8625,24.6108,28.2429,24.3266,24.8517,28.3778,25.7584,28.2986,20.8313,39.6717,25.4201,23.6175,27.4729,29.3216,30.3533,23.0691,30.6056,22.4833,26.9374,29.3124,24.2523,21.8616,30.2431,26.5288,22.5295,21.3865,26.789,27.8805,26.9023,22.4312,23.2832,24.9142,26.5098,24.1627,27.3494,27.9367,25.4093,27.0409,29.8673,26.2971,25.9965,23.3632,25.7873,28.5628,23.6298,33.9663,28.9743,25.2654,30.7504,24.1161,24.2581,26.9345,26.2376,29.2335,26.7177,25.8725,24.2699,24.6702,25.1203,30.9069,26.1547,24.4434,27.5442,25.3855,23.8318,26.3644,26.3519,24.2529,29.5388,25.6854,24.4125,26.3838,32.366,21.9373,34.4342,22.4929,21.9521,25.0662,22.3028,29.282,22.3988,25.3802,25.5797,25.7172,28.0195,25.1059,24.7473,21.6904,32.6387,22.7825,27.3092,24.1146,26.7346,26.1819,25.0805,25.9747,29.1962,22.6156,21.8795,25.6687,20.5385,25.3487,22.6313,22.6247,31.1125,25.081,24.5314,25.6894,24.6801,24.7036,25.373,21.0868,22.2456,25.4296,22.5005,25.4024,23.8454,24.7674,26.4613,26.4794,23.528,24.3599,24.0556,30.1035,24.3403,21.4013,23.4966,24.8746,25.7425,26.1238,27.9617,27.4528,25.0542,25.6323,27.657,22.5186,26.8681,23.9661,26.4975,28.1409,24.2494};
	static double [] oMin = {-24.752,-27.1414,-27.8564,-25.4112,-22.8066,-27.0455,-22.5373,-30.1219,-29.0846,-23.2653,-21.6596,-27.5099,-26.248,-26.2571,-27.6811,-25.7669,-24.7953,-21.7303,-21.2412,-21.2073,-26.9535,-23.6942,-23.7851,-29.6766,-30.4315,-21.3037,-22.1066,-26.5604,-29.5373,-21.6797,-26.1483,-29.4345,-24.017,-20.9091,-27.1311,-23.9236,-31.3359,-30.1133,-28.9992,-27.0379,-27.4457,-25.745,-24.1218,-28.9595,-24.2785,-24.0327,-31.5983,-27.2988,-27.9198,-28.4274,-24.9779,-25.3425,-24.2076,-24.6288,-25.7332,-26.429,-22.594,-25.073,-27.2818,-27.6674,-23.6567,-24.6312,-27.7886,-26.1847,-21.7761,-25.7602,-24.4424,-28.447,-23.1552,-25.4628,-30.5192,-25.0096,-28.8912,-24.4807,-24.065,-23.1486,-31.4696,-26.4372,-25.092,-27.3201,-24.2116,-21.2328,-26.1322,-32.4661,-22.8733,-24.4587,-30.7644,-24.67,-33.0042,-25.7868,-19.5824,-20.9858,-28.1496,-25.2628,-25.0596,-23.195,-23.6384,-26.5472,-24.1799,-23.9139,-22.0506,-27.2882,-21.7749,-27.071,-25.5557,-26.5141,-26.3089,-27.1942,-23.8693,-25.0837,-21.8454,-22.8583,-22.9794,-23.38,-28.9646,-24.02,-27.269,-23.9359,-27.2679,-23.9695,-25.5562,-28.1902,-27.1189,-22.2708,-26.9283,-28.4334,-25.6102,-21.276,-24.4433,-24.1858,-25.0128,-20.7569,-27.6686,-19.2397,-22.1651,-26.4415,-27.1275,-23.2719,-25.1942,-26.6631,-29.4843,-21.068,-24.6016,-26.4329,-24.8955,-27.0726,-26.0189,-25.1214,-25.4012,-21.1313,-27.0457,-28.3929,-22.9346,-25.8271,-25.2662,-19.9775,-26.9781,-22.2698,-25.2492,-26.2917,-26.4075,-27.3063,-28.8787,-26.6181,-25.5809,-29.3498,-22.7672,-28.3325,-25.4987,-24.2527,-26.272,-26.3582,-24.5538,-24.9961,-23.3946,-23.95,-26.1941,-31.1327,-25.5265,-25.3848,-30.4818,-26.7533,-25.2883,-24.6277,-31.1772,-24.5378,-26.65,-24.4572,-23.1354,-25.0667,-23.9245,-27.7164,-28.5907,-28.1013,-27.0361,-23.0817,-24.1862,-30.1406,-26.8778,-23.0996};
	
	boolean binaryNormalize = false;
	
	boolean preLoad = false;
	
	Random  rd = new Random(System.currentTimeMillis());
	
	boolean useTlengthShuffle = false;	

    public boolean isUseTlengthShuffle() {
        return useTlengthShuffle;
    }

    public void setUseTlengthShuffle(boolean useTlengthShuffle) {
        this.useTlengthShuffle = useTlengthShuffle;
    }
    
    List<IDataMatrix> dmCache = new ArrayList<IDataMatrix>();
    
    public void preLoad() {
    	preLoad = true;
    	for (int i = 0; i < imgLoader.imgs.size(); i++) {
    		String s = imgLoader.imgs.get(i);
    		IDataMatrix dm = constructIDataMatrix(s, imgLoader.header.startsWith(label));
    		dmCache.add(dm);
		}
	}
    
    public IDataMatrix getIDataMatrix(int id) {
		if (preLoad && dmCache.size() > 0) {
			return dmCache.get(id);
		} else {
			String s = imgLoader.imgs.get(id);
			boolean hasHeader = false;
	        if (imgLoader.header != null) {
	            hasHeader = imgLoader.header.startsWith(label);
	        }
			return constructIDataMatrix(s, hasHeader);
		}
	}

    @Override
	public IDataMatrix next() {
		cnt++;
		double l = imgLoader.imgs.size();
		double segment = l;
		double base = 0;
		if (useTlengthShuffle) {
		    base = l/(double)tLength * (double)(cnt % tLength);
            segment = l/(double)tLength;
        }
		
		int ri  = (int) (rd.nextDouble() * segment + base);
		if (ri == l) {//exclusive, so no need to worry about it.
			ri = ri - 1;
		}
//		String s = imgLoader.imgs.get(ri);
//		return constructIDataMatrix(s, imgLoader.header.startsWith(label));
		return getIDataMatrix(ri);
	}
    
    public IDataMatrix next(Object pos) {
		cnt++;
		int ri  = (Integer) pos;
		return getIDataMatrix(ri);
	}
	
	int rLength;
	
	int wordsNum = 1;
	
	public IDataMatrix constructIDataMatrix(String s, boolean h) {
		DataMatrix dataMatrix = new DataMatrix();
		String [] arr = s.split(splitter);
		double [] ta = new double[tLength];
		int cnt = 0;
		int ml = 0;
		int aLength = 0; 
		if (h) {
			int t = Integer.parseInt(arr[cnt ++]);
			ta[t - 1] = 1;
//			ml = (int) Math.sqrt(arr.length - 1);
			dataMatrix.setTarget(ta);
			aLength = arr.length - 1;
		} else {
//			ml = (int) Math.sqrt(arr.length);
			dataMatrix.setTarget(null);
			aLength = arr.length;
		}
		if ((aLength)/rLength < wordsNum) {
            return null;
        }
		int r = (aLength)/rLength;
//		r = 20;
		double [][] m = new double[r][rLength];
		for (int i = 0; i < m.length; i++) {
			m[i] = new double[rLength];
			for (int j = 0; j < m[i].length; j++) {			    
				double a = 0;
				if (cnt < arr.length) {
                    a = Double.parseDouble(arr[cnt ++]);
                }				
//				sum = sum + a/omax;
//				if (binaryNormalize) {
//					if (a > 0) {
//						m[i][j] = max;
//					} else {
//						m[i][j] = min;
//					}
//				} else {
					m[i][j] = (a - omin)/(omax - omin) *
						(max - min) + min;
//				}
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
	
	public W2VDataStreamV2(CsvImgLoader imgLoader, int tLength, int rLength) {
		super();
		this.imgLoader = imgLoader;
		this.tLength = tLength;
		this.rLength = rLength;
	}

	@Override
	public boolean hasNext() {
		if (cnt <= imgLoader.imgs.size() - 1) {
			return true;
		}
		return false;
	}
	
	public static void main(String[] args) throws Exception {
		String file = "E://models//cnn//sentiment3.0//cnn-sentiment-training-26000.txt";
		CsvImgLoader imgLoader = new CsvImgLoader();
		imgLoader.loadImg(file);
		int kLength = 3; //类别数
		int cnt = 0;
		W2VDataStreamV2 w2vDataStream = new W2VDataStreamV2(imgLoader, kLength, 200);
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
