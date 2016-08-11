package deepDriver.dl.aml.cnn.test;

import java.io.File;


import deepDriver.dl.aml.ann.ANNCfg;
import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.cnn.ActivationFactory;
import deepDriver.dl.aml.cnn.CNNArchitecture;
import deepDriver.dl.aml.cnn.CNNConfigurator;
import deepDriver.dl.aml.cnn.ConvolutionNeuroNetwork;
import deepDriver.dl.aml.cnn.LayerConfigurator;
import deepDriver.dl.aml.cnn.LeakyReLU;
import deepDriver.dl.aml.cnn.img.CsvImgLoader;
import deepDriver.dl.aml.cnn.img.ImgDataStream;

public class TestHwrCNN {
	
	public void test(String mfile, String file, String tfile) throws Exception {
		System.out.println("Load training file from "+file);
		CsvImgLoader imgLoader = new CsvImgLoader();
		imgLoader.loadImg(file);
		int kLength = 10;
		ImgDataStream imgDataStream = new ImgDataStream(imgLoader, kLength);
		
		System.out.println("Load testing file from "+tfile);
		CsvImgLoader timgLoader = new CsvImgLoader();
		timgLoader.loadImg(tfile);
		ImgDataStream timgDataStream = new ImgDataStream(timgLoader, kLength);
		
		ConvolutionNeuroNetwork cnn = new ConvolutionNeuroNetwork();
		cnn.readCfg(mfile);
//		cnn.enableTest(true);
		
//		cnn.test(imgDataStream);
//		System.out.println("Done with testing training file");
		
		cnn.test(timgDataStream);
//		cnn.enableTest(false);
		System.out.println("Done with testing testing file");
	}
	
	public void train(String file, String tfile) throws Exception {
		System.out.println("Load training file from "+file);
		CsvImgLoader imgLoader = new CsvImgLoader();
		imgLoader.loadImg(file);
		int kLength = 10;
		ImgDataStream imgDataStream = new ImgDataStream(imgLoader, kLength);
//		imgDataStream.preLoad();
		
		System.out.println("Load testing file from "+tfile);
		CsvImgLoader timgLoader = new CsvImgLoader();
		timgLoader.loadImg(tfile);
		ImgDataStream timgDataStream = new ImgDataStream(timgLoader, kLength);
		
		CNNConfigurator cnnCfg = new CNNConfigurator(); 
		cnnCfg.setL(0.001);
		cnnCfg.setName("hwr");
		CNNArchitecture ca = new CNNArchitecture(); 
		
//		LeakyReLU acf = new LeakyReLU();
//		acf.setA(0.001); 
		IActivationFunction acf = null; //ActivationFactory.getAf().getReLU();
		
		LayerConfigurator lc0 = new LayerConfigurator(LayerConfigurator.CONVOLUTION_LAYER, 
				1, true, 28, 28, 1);
		lc0.setAcf(acf);
//		lc0.setFMAdaptive(true);
		ca.addLayerCfg(lc0);
		
		LayerConfigurator lc1 = new LayerConfigurator(LayerConfigurator.CONVOLUTION_LAYER, 
				6, true, 5, 5, 1);
		lc1.setAcf(acf);
//		lc1.setFMAdaptive(true);
		/**different cks
		lc1.setCks(new int[][]{
				{2, 200},{2, 200},{2, 200},
				{3, 200},{3, 200},{3, 200}
		});****/
		ca.addLayerCfg(lc1);
		
		LayerConfigurator lc2 = new LayerConfigurator(LayerConfigurator.POOLING_LAYER, 
				6, true, 2, 2, 1);
		lc2.setAcf(acf);
//		lc2.setCKAdaptive(true);
		ca.addLayerCfg(lc2);
		
		LayerConfigurator lc3 = new LayerConfigurator(LayerConfigurator.CONVOLUTION_LAYER, 
				16, false, 5, 5, 1);
		lc3.setAcf(acf);
		int [][] lc3_fam = new int[][] {
				{1, 1, 1, 0, 0, 0},
				{0, 1, 1, 1, 0, 0},
				{0, 0, 1, 1, 1, 0},
				{0, 0, 0, 1, 1, 1},
				{1, 0, 0, 0, 1, 1},
				{1, 1, 0, 0, 0, 1},
				
				{1, 1, 1, 1, 0, 0},
				{0, 1, 1, 1, 1, 0},
				{0, 0, 1, 1, 1, 1}, 
				{1, 0, 0, 1, 1, 1},
				{1, 1, 0, 0, 1, 1}, 
				{1, 1, 1, 0, 0, 1}, 
				
				{1, 1, 0, 1, 1, 0}, 
				{0, 1, 1, 0, 1, 1}, 
				{1, 0, 1, 1, 0, 1},
				{1, 1, 1, 1, 1, 1}
		};
		lc3.setFeatureMapAllocationMatrix(lc3_fam);
		ca.addLayerCfg(lc3);
		
		LayerConfigurator lc4 = new LayerConfigurator(LayerConfigurator.POOLING_LAYER, 
				16, true, 2, 2, 1);
		lc4.setAcf(acf);
		ca.addLayerCfg(lc4);
		
		LayerConfigurator lc5 = new LayerConfigurator(LayerConfigurator.CONVOLUTION_LAYER, 
				120, true, 4, 4, 1);
		lc5.setAcf(acf);
		ca.addLayerCfg(lc5);
		
		int threadNum = 2;
		
		ANNCfg aNNCfg = new ANNCfg();

		aNNCfg.setThreadsNum(threadNum);
		aNNCfg.setDropOut(0.12);

//		aNNCfg.setDropOut(0.1);
		aNNCfg.setDropOut(0);
		aNNCfg.setTesting(false);
		//aNNCfg.setTesting(true);
		
		IActivationFunction relu = ActivationFactory.getAf().getReLU();
		
		LayerConfigurator lc6 = new LayerConfigurator(LayerConfigurator.ANN_LAYER, 
				120, true, 1, 1, 1);
		lc6.setAcf(relu);
		ca.addLayerCfg(lc6);
		lc6.setaNNCfg(aNNCfg);
		
		LayerConfigurator lc7 = new LayerConfigurator(LayerConfigurator.ANN_LAYER, 
				84, true, 1, 1, 1);
		lc7.setAcf(relu);
		ca.addLayerCfg(lc7);
		lc7.setaNNCfg(aNNCfg);
		
		LayerConfigurator lc8 = new LayerConfigurator(LayerConfigurator.ANN_LAYER, 
				kLength, true, 1, 1, 1);
		lc8.setAcf(relu);
		lc8.setLast(true);
		ca.addLayerCfg(lc8);
		lc8.setaNNCfg(aNNCfg);
		
		cnnCfg.setUseBN(true);
		cnnCfg.setThreadsNum(threadNum);
		cnnCfg.setPoolingType(CNNConfigurator.MAX_POOLING_TYPE);
		ConvolutionNeuroNetwork cnn = new ConvolutionNeuroNetwork();
		cnn.construct(ca, cnnCfg);
		cnn.setDebug(true);
		cnn.train(imgDataStream, timgDataStream);
		System.out.println("Done with training.");
	//	cnn.test(timgDataStream);
	}
	
	public static void main(String[] args) throws Exception {
		TestHwrCNN testHwrCNN = new TestHwrCNN();

		String sf = "E:\\data\\CNN\\";

		File fsf = new File(sf);
		if (!fsf.exists()) {			
			sf = System.getProperty("user.dir");
		}			
		File dir = new File(sf, "data");
		dir.mkdirs();


		
//		testHwrCNN.train(dir.getAbsolutePath()+"\\train.csv", 
//				dir.getAbsolutePath()+"\\test-new-2.csv");
		
		testHwrCNN.train(dir.getAbsolutePath()+"\\kaggleTest\\modelTrain", 
		dir.getAbsolutePath()+"\\kaggleTest\\modelTest");
//		testHwrCNN.test(dir.getAbsolutePath()+"\\cnnCfg_hwr-dropout0.12-400.m",
//				dir.getAbsolutePath()+"\\train-new-2.csv", 
//				dir.getAbsolutePath()+"\\test.csv");
		
				//dir.getAbsolutePath()+"\\modelTest");



	}

}
