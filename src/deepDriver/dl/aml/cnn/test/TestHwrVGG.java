package deepDriver.dl.aml.cnn.test;

import java.io.File;


import deepDriver.dl.aml.ann.ANNCfg;
import deepDriver.dl.aml.cnn.CNNArchitecture;
import deepDriver.dl.aml.cnn.CNNConfigurator;
import deepDriver.dl.aml.cnn.ConvolutionNeuroNetwork;
import deepDriver.dl.aml.cnn.LayerConfigurator;
import deepDriver.dl.aml.cnn.LeakyReLU;
import deepDriver.dl.aml.cnn.img.CsvImgLoader;
import deepDriver.dl.aml.cnn.img.ImgDataStream;

public class TestHwrVGG {
    
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
//      cnn.enableTest(true);
        
//      cnn.test(imgDataStream);
//      System.out.println("Done with testing training file");
        
        cnn.test(timgDataStream);
//      cnn.enableTest(false);
        System.out.println("Done with testing testing file");
    }
    
    
    
    
    
	public void train(String file, String tfile) throws Exception {
		System.out.println("Load training file from "+file);
		CsvImgLoader imgLoader = new CsvImgLoader();
		imgLoader.loadImg(file);
		int kLength = 10;
		ImgDataStream imgDataStream = new ImgDataStream(imgLoader, kLength);
		
		System.out.println("Load testing file from "+tfile);
		CsvImgLoader timgLoader = new CsvImgLoader();
		timgLoader.loadImg(tfile);
		ImgDataStream timgDataStream = new ImgDataStream(timgLoader, kLength);
		
		CNNConfigurator cnnCfg = new CNNConfigurator(); 
		cnnCfg.setL(0.001);
		cnnCfg.setName("hwr4VGG");
		CNNArchitecture ca = new CNNArchitecture(); 
		
//		LeakyReLU acf = new LeakyReLU();
//		acf.setA(0.001); 
		LeakyReLU acf = null;
		
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
		
		LayerConfigurator lc11 = new LayerConfigurator(LayerConfigurator.CONVOLUTION_LAYER, 
				16, true, 3, 3, 1);
		lc11.setAcf(acf);
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
		
		LayerConfigurator lc31 = new LayerConfigurator(LayerConfigurator.CONVOLUTION_LAYER, 
				32, true, 3, 3, 1);
		lc31.setAcf(acf);
		ca.addLayerCfg(lc31);
		
		LayerConfigurator lc4 = new LayerConfigurator(LayerConfigurator.POOLING_LAYER, 
				32, true, 2, 2, 1);
		lc4.setAcf(acf);
		lc4.setPadding(1);
		ca.addLayerCfg(lc4);
		
		LayerConfigurator lc5 = new LayerConfigurator(LayerConfigurator.CONVOLUTION_LAYER, 
				32, true, 3, 3, 1);
		lc5.setAcf(acf);
		ca.addLayerCfg(lc5);
		
		LayerConfigurator lc51 = new LayerConfigurator(LayerConfigurator.POOLING_LAYER, 
				32, true, 2, 2, 1);
		lc51.setAcf(acf);
		lc51.setPadding(1);
		ca.addLayerCfg(lc51);
		
		LayerConfigurator lc511 = new LayerConfigurator(LayerConfigurator.CONVOLUTION_LAYER, 
				32, true, 3, 3, 1);
		lc511.setAcf(acf);
		ca.addLayerCfg(lc511);
		
		LayerConfigurator lc512 = new LayerConfigurator(LayerConfigurator.POOLING_LAYER, 
				32, true, 2, 2, 1);
		lc512.setAcf(acf);
		lc512.setPadding(1);
		ca.addLayerCfg(lc512);
		
		ANNCfg aNNCfg = new ANNCfg();
		aNNCfg.setDropOut(0.1);
		aNNCfg.setTesting(false);
		
		LayerConfigurator lc6 = new LayerConfigurator(LayerConfigurator.ANN_LAYER, 
				128, true, 1, 1, 1);
		lc6.setAcf(acf);
		ca.addLayerCfg(lc6);
		lc6.setaNNCfg(aNNCfg);
		
		LayerConfigurator lc7 = new LayerConfigurator(LayerConfigurator.ANN_LAYER, 
				64, true, 1, 1, 1);
		lc7.setAcf(acf);
		ca.addLayerCfg(lc7);
		lc7.setaNNCfg(aNNCfg);
		
		LayerConfigurator lc8 = new LayerConfigurator(LayerConfigurator.ANN_LAYER, 
				kLength, true, 1, 1, 1);
		lc8.setAcf(acf);
		lc8.setLast(true);
		ca.addLayerCfg(lc8);
		lc8.setaNNCfg(aNNCfg);
		
		cnnCfg.setPoolingType(CNNConfigurator.MAX_POOLING_TYPE);
		ConvolutionNeuroNetwork cnn = new ConvolutionNeuroNetwork();
		cnn.construct(ca, cnnCfg);
		cnn.train(imgDataStream, timgDataStream);
		System.out.println("Done with training.");
		cnn.test(timgDataStream);
	}
	
	public static void main(String[] args) throws Exception {
		TestHwrVGG testHwrCNN = new TestHwrVGG();
		String sf = "E:\\data\\CNN\\";
		File fsf = new File(sf);
		if (!fsf.exists()) {			
			sf = System.getProperty("user.dir");
		}			
		File dir = new File(sf, "data");
		dir.mkdirs();
//		testHwrCNN.train(dir.getAbsolutePath()+"\\kaggleTest\\modelTrain", 
//				dir.getAbsolutePath()+"\\kaggleTest\\modelTest");
		testHwrCNN.test(dir.getAbsolutePath()+"\\cnnCfg_1463491591678_hwr4VGG.m",
				dir.getAbsolutePath()+"\\train.csv", 
				dir.getAbsolutePath()+"\\test.csv");
	}
}
