package deepDriver.dl.aml.cnn.test;

import java.io.File;


import deepDriver.dl.aml.cnn.IDataMatrix;
import deepDriver.dl.aml.cnn.img.CsvImgLoader;
import deepDriver.dl.aml.cnn.img.W2VDataStream;

public class TxtAnalyzer {
	
	public static void main(String[] args) throws Exception {
		String sf = "D:\\6.workspace\\ANN\\cnn";
		int kLength = 3;
		int width = 200;

        File fsf = new File(sf);
        if (!fsf.exists()) {            
            sf = System.getProperty("user.dir");
        }           
        File dir = new File(sf, "sentiment3.0");
        dir.mkdirs();
        String fname = "6000training_V3.csv";
		String file = dir.getAbsolutePath()+"\\"+fname;
		System.out.println("Load training file from "+file);
		CsvImgLoader w2vLoader = new CsvImgLoader();
		w2vLoader.loadImg(file);
		W2VDataStream trainingDs = new W2VDataStream(w2vLoader, kLength, width);
		DataMetrics [] dataMetrics = new DataMetrics[width]; 
		
		double max = 0;
		double min = 0;
		while (trainingDs.hasNext()) {
			IDataMatrix dm = trainingDs.next();
			if (dm == null) {
				continue;
			}
			double [][] m = dm.getMatrix();
			for (int i = 0; i < m.length; i++) {
				for (int j = 0; j < m[i].length; j++) {
					if (dataMetrics[j] == null) {
						dataMetrics[j] = new DataMetrics();
					}
					if (dataMetrics[j].max < m[i][j]) {
						dataMetrics[j].max = m[i][j];
					}
					if (dataMetrics[j].min > m[i][j]) {
						dataMetrics[j].min = m[i][j];
					} 
					dataMetrics[j].sum = dataMetrics[j].sum + m[i][j];
					dataMetrics[j].cnt ++;
					
					if (max < m[i][j]) {
						max = m[i][j];
					}
					if (min > m[i][j]) {
						min = m[i][j];
					}
				}
			}
		}
		
		System.out.println("Max value is: "+max);
		System.out.println("Min value is: "+min);
		
		System.out.println("Max are: ");		
		for (int i = 0; i < dataMetrics.length; i++) {
			System.out.print(dataMetrics[i].max+",");
		}
		System.out.println();
		
		System.out.println("Min are: ");
		for (int i = 0; i < dataMetrics.length; i++) {
			System.out.print(dataMetrics[i].min+",");
		}
		System.out.println();
		
		System.out.println("Avg are: ");
		for (int i = 0; i < dataMetrics.length; i++) {
			dataMetrics[i].avg = (dataMetrics[i].sum/(double)dataMetrics[i].cnt);
			System.out.print(dataMetrics[i].avg+",");
		}
		System.out.println();
		
		
		System.out.println();
		
	}

}
