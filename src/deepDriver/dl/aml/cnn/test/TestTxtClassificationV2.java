package deepDriver.dl.aml.cnn.test;

import java.io.File;


import deepDriver.dl.aml.ann.ANNCfg;
import deepDriver.dl.aml.cnn.CNNArchitecture;
import deepDriver.dl.aml.cnn.CNNConfigurator;
import deepDriver.dl.aml.cnn.ConvolutionNeuroNetwork;
import deepDriver.dl.aml.cnn.IDataStream;
import deepDriver.dl.aml.cnn.LayerConfigurator;
import deepDriver.dl.aml.cnn.LeakyReLU;
import deepDriver.dl.aml.cnn.img.CsvImgLoader;
import deepDriver.dl.aml.cnn.img.ImgDataStream;
import deepDriver.dl.aml.cnn.img.W2VDataStream;

public class TestTxtClassificationV2 {
	
	public void train(String file, String tfile) throws Exception {
		
		int kLength = 3;
		int fixedRow = 20;
		
		System.out.println("Load training file from "+file);
		CsvImgLoader w2vLoader = new CsvImgLoader();
		w2vLoader.loadImg(file);
		W2VDataStream trainingDs = new W2VDataStream(w2vLoader, kLength, 200);
		trainingDs.setFixedRow(fixedRow);
		
		System.out.println("Load testing file from "+ tfile);
        CsvImgLoader tw2vLoader = new CsvImgLoader();
        tw2vLoader.loadImg(tfile);
        W2VDataStream testDs = new W2VDataStream(tw2vLoader, kLength, 200);
        testDs.setFixedRow(fixedRow);
        
		CNNConfigurator cnnCfg = new CNNConfigurator(); 
		cnnCfg.setL(0.001);
		cnnCfg.setName("TxtCf");
		CNNArchitecture ca = new CNNArchitecture(); 
		
//		LeakyReLU acf = new LeakyReLU();
//		acf.setA(0.001);
		LeakyReLU acf = null;
		
		LayerConfigurator lc0 = new LayerConfigurator(LayerConfigurator.CONVOLUTION_LAYER, 
				1, true, 20, 200, 1);
		lc0.setAcf(acf);
		ca.addLayerCfg(lc0);
		
		LayerConfigurator lc1 = new LayerConfigurator(LayerConfigurator.CONVOLUTION_LAYER, 
		        100, true, 2, 200, 1);
		lc1.setAcf(acf);
//		lc1.setFMAdaptive(true);
		lc1.setCks(new int[][]{
                {2, 200},{2, 200},{2, 200},{2, 200},{2, 200},{2, 200},
                {3, 200},{3, 200},{3, 200},{3, 200},{3, 200},{3, 200},
                {2, 200},{2, 200},{2, 200},{2, 200},{2, 200},{2, 200},
                {3, 200},{3, 200},{3, 200},{3, 200},{3, 200},{3, 200},
                {2, 200},{2, 200},{2, 200},{2, 200},{2, 200},{2, 200},
                {3, 200},{3, 200},{3, 200},{3, 200},{3, 200},{3, 200},
                {2, 200},{2, 200},{2, 200},{2, 200},{2, 200},{2, 200},
                {3, 200},{3, 200},{3, 200},{3, 200},{3, 200},{3, 200},
                {2, 200},{2, 200},{2, 200},{2, 200},{2, 200},{2, 200},
                {3, 200},{3, 200},{3, 200},{3, 200},{3, 200},{3, 200},
                
                {2, 200},{2, 200},{2, 200},{2, 200},{2, 200},{2, 200},
                {3, 200},{3, 200},{3, 200},{3, 200},{3, 200},{3, 200},
                {2, 200},{2, 200},{2, 200},{2, 200},{2, 200},{2, 200},
                {3, 200},{3, 200},{3, 200},{3, 200},{3, 200},{3, 200},
                {2, 200},{2, 200},{2, 200},{2, 200},{2, 200},{2, 200},
                {3, 200},{3, 200},{3, 200},{3, 200},{3, 200},{3, 200},
                {2, 200},{2, 200},
                {3, 200},{3, 200},
                
        });
		ca.addLayerCfg(lc1);
		
		LayerConfigurator lc2 = new LayerConfigurator(LayerConfigurator.POOLING_LAYER, 
		        100, true, 19, 1, 1);
		lc2.setAcf(acf);
//		lc2.setCKAdaptive(true);
		ca.addLayerCfg(lc2);
		
		ANNCfg aNNCfg = new ANNCfg();
		aNNCfg.setDropOut(-1);
		aNNCfg.setTesting(false);
		
		LayerConfigurator lc6 = new LayerConfigurator(LayerConfigurator.ANN_LAYER, 
		        100, true, 1, 1, 1);
		lc6.setAcf(acf);
		ca.addLayerCfg(lc6);
		lc6.setaNNCfg(aNNCfg);
		
		LayerConfigurator lc7 = new LayerConfigurator(LayerConfigurator.ANN_LAYER, 
				60, true, 1, 1, 1);
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
		cnn.train(trainingDs, testDs);
//		cnn.test(trainingDs);
		System.out.println("Done with training.");
	}
	
	
	
	
	public void test(String mfile, String file, String tfile) throws Exception {
        
	    int kLength = 3;
	    
//	    System.out.println("Load training file from "+file);
//      CsvImgLoader trainingLoader = new CsvImgLoader();
//      trainingLoader.loadImg(file);
//      W2VDataStream trainingStream = new W2VDataStream(trainingLoader, kLength, 200);
        
        
        System.out.println("Load testing file from "+tfile);
        CsvImgLoader testLoader = new CsvImgLoader();
        testLoader.loadImg(tfile);
        W2VDataStream testStream = new W2VDataStream(testLoader, kLength, 200);
        
        ConvolutionNeuroNetwork cnn = new ConvolutionNeuroNetwork();
        cnn.readCfg(mfile);
        cnn.enableTest(true);
        
        
        cnn.test(testStream);
        cnn.enableTest(false);
        System.out.println("Done with testing testing file");
    }
	
	public static void main(String[] args) throws Exception 
	{
	    TestTxtClassificationV2 testTxtClassification = new TestTxtClassificationV2();

        String sf = "E:\\models\\CNN\\";

        File fsf = new File(sf);
        if (!fsf.exists()) {            
            sf = System.getProperty("user.dir");
        }           
        File dir = new File(sf, "sentiment3.0");
        dir.mkdirs();

        testTxtClassification.train(dir.getAbsolutePath()+"\\test2.csv", dir.getAbsolutePath()+"\\test2.csv");
        
//        testTxtClassification.test(dir.getAbsolutePath()+"\\cnnCfg_hwr-dropout0.12-400.m",
//                dir.getAbsolutePath()+"\\train-new-2.csv", 
//                dir.getAbsolutePath()+"\\test.csv");
        
    }
	

}
