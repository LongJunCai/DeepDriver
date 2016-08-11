package deepDriver.dl.aml.dnn.test;

import java.io.File;


import deepDriver.dl.aml.ann.InputParameters;
import deepDriver.dl.aml.cnn.img.CsvImgLoader;
import deepDriver.dl.aml.cnn.img.ImgDataStream;
import deepDriver.dl.aml.costFunction.SoftMax4ANN;
import deepDriver.dl.aml.dnn.DNN;

public class TestDNN4Hwr {	
	
	public void train(String file, String tfile) throws Exception {
		int kLength = 10;
		CsvImgLoader imgLoader = new CsvImgLoader();
		imgLoader.loadImg(file);
		ImgDataStream is = new ImgDataStream(imgLoader, kLength);
		
		StreamAdapter streamAdapter = new StreamAdapter();
		InputParameters ip = new InputParameters();
		streamAdapter.loadFromStream(is, ip);
		
		ip.setIterationNum(3000);
		ip.setLamda(0.0000001);
		ip.setLayerNum(8);
		ip.setNeuros(new int[]{90, 90, 90, 90, 90, 90, 90, kLength});
		
		DNN dnn = new DNN();
		dnn.setCf(new SoftMax4ANN());
		dnn.setkLength(kLength);
		dnn.getaNNCfg().setDropOut(0);
		
		dnn.trainModel(ip);
	}
	
	public static void main(String[] args) throws Exception {
		TestDNN4Hwr test = new TestDNN4Hwr();
		String sf = "E:\\0.workspace\\4.data\\cnn";
		File fsf = new File(sf);
		if (!fsf.exists()) {			
			sf = System.getProperty("user.dir");
		}			
		File dir = new File(sf, "data");
		dir.mkdirs();
		test.train(dir.getAbsolutePath()+"\\kaggleTest\\modelTrain", 
				dir.getAbsolutePath()+"\\kaggleTest\\modelTest");
//		test.train(String file, String tfile);
		
		
	}

}
