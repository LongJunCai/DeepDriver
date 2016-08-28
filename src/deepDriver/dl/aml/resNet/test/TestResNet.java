package deepDriver.dl.aml.resNet.test;

import java.io.File;

import deepDriver.dl.aml.ann.ANNCfg;
import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.cnn.CNNArchitecture;
import deepDriver.dl.aml.cnn.CNNConfigurator;
import deepDriver.dl.aml.cnn.LayerConfigurator;
import deepDriver.dl.aml.cnn.img.CsvImgLoader;
import deepDriver.dl.aml.cnn.img.ImgDataStream;
import deepDriver.dl.aml.fnn.FractalNet;
import deepDriver.dl.aml.cnn.ActivationFactory;


public class TestResNet {
	

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
		
		int threadsNum = 4;
		CNNConfigurator cnnCfg = new CNNConfigurator(); 
		cnnCfg.setUseBN(true);//?
		cnnCfg.setThreadsNum(threadsNum);
		cnnCfg.setL(0.001);
		cnnCfg.setName("hwr4fnn");
		CNNArchitecture ca = new CNNArchitecture(); 
		
//		LeakyReLU acf = new LeakyReLU();
//		acf.setA(0.001); 
		IActivationFunction acf = null;//ActivationFactory.getAf().getTanh();
		
		LayerConfigurator lc0 = new LayerConfigurator(LayerConfigurator.CONVOLUTION_LAYER, 
				1, true, 28, 28, 1);
		lc0.setAcf(acf);
		lc0.setPadding(1);
//		lc0.setFMAdaptive(true);
		ca.addLayerCfg(lc0);
		
//		LayerConfigurator lc1 = new LayerConfigurator(LayerConfigurator.CONVOLUTION_LAYER, 
//				16, true, 3, 3, 1);
//		lc1.setAcf(acf);
//		lc1.setPadding(1);
//		ca.addLayerCfg(lc1);
		
		int fractalDepth = 2;	
		int blNum = 3;
		boolean resNetLayer = false;
		
		LayerConfigurator lc11 = new LayerConfigurator(LayerConfigurator.FRACTAL_BLOCK_LAYER, 
				16, true, 3, 3, 1);
		lc11.setAcf(acf);
		lc11.setResNetLayer(resNetLayer);
		lc11.setFblockLayerNum(blNum);
		lc11.setFblockDepth(fractalDepth);
		lc11.setPadding(1);
		ca.addLayerCfg(lc11);
		
		LayerConfigurator lc2 = new LayerConfigurator(LayerConfigurator.POOLING_LAYER, 
				16, true, 2, 2, 1);
		lc2.setAcf(acf);
		lc2.setPadding(1);
		ca.addLayerCfg(lc2);
		
//		LayerConfigurator lc3 = new LayerConfigurator(LayerConfigurator.CONVOLUTION_LAYER, 
//				32, true, 3, 3, 1);
//		lc3.setAcf(acf);
//		lc3.setPadding(1);
//		ca.addLayerCfg(lc3);
		
		LayerConfigurator lc31 = new LayerConfigurator(LayerConfigurator.FRACTAL_BLOCK_LAYER, 
				32, true, 3, 3, 1);
		lc31.setAcf(acf);
		lc31.setFblockLayerNum(blNum);
		lc31.setResNetLayer(resNetLayer);
		lc31.setFblockDepth(fractalDepth);
		lc31.setPadding(1);
		ca.addLayerCfg(lc31);
		
		LayerConfigurator lc4 = new LayerConfigurator(LayerConfigurator.POOLING_LAYER, 
				32, true, 2, 2, 1);
		lc4.setAcf(acf);
		lc4.setPadding(1);
		ca.addLayerCfg(lc4);
		
		LayerConfigurator lc5 = new LayerConfigurator(LayerConfigurator.FRACTAL_BLOCK_LAYER, 
				64, true, 3, 3, 1);
		lc5.setAcf(acf);
		lc5.setFblockLayerNum(blNum);
		lc5.setResNetLayer(resNetLayer);
		lc5.setFblockDepth(fractalDepth);
		lc5.setPadding(1);
		ca.addLayerCfg(lc5);
		
		LayerConfigurator lc51 = new LayerConfigurator(LayerConfigurator.POOLING_LAYER, 
				64, true, 2, 2, 1);
		lc51.setAcf(acf);
		lc51.setPadding(1);
		ca.addLayerCfg(lc51);
		
		LayerConfigurator lc511 = new LayerConfigurator(LayerConfigurator.FRACTAL_BLOCK_LAYER, 
				128, true, 3, 3, 1);
		lc511.setAcf(acf);
		lc511.setFblockLayerNum(blNum);
		lc511.setResNetLayer(resNetLayer);
		lc511.setFblockDepth(fractalDepth);
		lc511.setPadding(1);
		ca.addLayerCfg(lc511);
		
		LayerConfigurator lc512 = new LayerConfigurator(LayerConfigurator.POOLING_LAYER, 
				128, true, 2, 2, 1);
		lc512.setAcf(acf);
		lc512.setPadding(1);
		ca.addLayerCfg(lc512);
		
		LayerConfigurator lc521 = new LayerConfigurator(LayerConfigurator.FRACTAL_BLOCK_LAYER, 
				256, true, 3, 3, 1);
		lc521.setAcf(acf);
		lc521.setFblockLayerNum(blNum);
		lc521.setResNetLayer(resNetLayer);
		lc521.setFblockDepth(fractalDepth);
		lc521.setPadding(1);
		ca.addLayerCfg(lc521);
//		
		LayerConfigurator lc522 = new LayerConfigurator(LayerConfigurator.POOLING_LAYER, 
				256, true, 2, 2, 1);
		lc522.setAcf(acf);
		lc522.setPadding(1);
		ca.addLayerCfg(lc522);		
		
		ANNCfg aNNCfg = new ANNCfg();
		aNNCfg.setDropOut(0);
		aNNCfg.setTesting(false);
		aNNCfg.setThreadsNum(threadsNum);
		
		IActivationFunction relu = ActivationFactory.getAf().getReLU();
		
		LayerConfigurator lc6 = new LayerConfigurator(LayerConfigurator.ANN_LAYER, 
				256, true, 1, 1, 1);
		lc6.setAcf(relu);
		ca.addLayerCfg(lc6);
		lc6.setaNNCfg(aNNCfg);
		
		LayerConfigurator lc7 = new LayerConfigurator(LayerConfigurator.ANN_LAYER, 
				64, true, 1, 1, 1);
		lc7.setAcf(relu);
		ca.addLayerCfg(lc7);
		lc7.setaNNCfg(aNNCfg);
		
		LayerConfigurator lc8 = new LayerConfigurator(LayerConfigurator.ANN_LAYER, 
				kLength, true, 1, 1, 1);
		lc8.setAcf(relu);
		lc8.setLast(true);
		ca.addLayerCfg(lc8);
		lc8.setaNNCfg(aNNCfg);
		
		cnnCfg.setPoolingType(CNNConfigurator.MAX_POOLING_TYPE);
		FractalNet fnn = new FractalNet();
		fnn.setDebug(true);
		fnn.construct(ca, cnnCfg);
		fnn.train(imgDataStream, timgDataStream);
		System.out.println("Done with training.");
		fnn.test(timgDataStream);
	}
	
	public static void main(String[] args) throws Exception {
		TestResNet resNet = new TestResNet();
		String sf = "D:\\6.workspace\\ANN\\cnn\\";
		File fsf = new File(sf);
		if (!fsf.exists()) {			
			sf = System.getProperty("user.dir");
		}			
		File dir = new File(sf, "data");
		dir.mkdirs();
		resNet.train(dir.getAbsolutePath()+"\\kaggleTest\\modelTrain", 
				dir.getAbsolutePath()+"\\kaggleTest\\modelTest");
//		testHwrCNN.test(dir.getAbsolutePath()+"\\cnnCfg_1463491591678_hwr4VGG.m",
//				dir.getAbsolutePath()+"\\kaggleTest\\modelTrain", 
//				dir.getAbsolutePath()+"\\kaggleTest\\modelTest");
	}

}
