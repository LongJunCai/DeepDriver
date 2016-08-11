package deepDriver.dl.aml.cnn.test;

import java.io.File;


import deepDriver.dl.aml.ann.ANNCfg;
import deepDriver.dl.aml.cnn.CNNArchitecture;
import deepDriver.dl.aml.cnn.CNNConfigurator;
import deepDriver.dl.aml.cnn.ConvolutionNeuroNetwork;
import deepDriver.dl.aml.cnn.LayerConfigurator;
import deepDriver.dl.aml.cnn.LeakyReLU;
import deepDriver.dl.aml.cnn.img.CsvImgLoader;
import deepDriver.dl.aml.cnn.img.W2VDataStream;
import deepDriver.dl.aml.cnn.img.W2VDataStreamV2;

public class TestTxtClassification {
	
	public void train(String file, String tfile) throws Exception {
		
		int kLength = 3;
//		int fixedRow = 20;
		
		System.out.println("Load training file from "+file);
		CsvImgLoader w2vLoader = new CsvImgLoader();
		w2vLoader.loadImg(file);
		W2VDataStreamV2 trainingDs = new W2VDataStreamV2(w2vLoader, kLength, 200);
		trainingDs.setUseTlengthShuffle(true);
		trainingDs.preLoad();
		//trainingDs.setFixedRow(fixedRow);
		
		System.out.println("Load testing file from "+ tfile);
        CsvImgLoader tw2vLoader = new CsvImgLoader();
        tw2vLoader.loadImg(tfile);
        W2VDataStreamV2 testDs = new W2VDataStreamV2(tw2vLoader, kLength, 200);
       // testDs.setFixedRow(fixedRow);
		
		CNNConfigurator cnnCfg = new CNNConfigurator(); 
		cnnCfg.setL(0.01);
		cnnCfg.setName("TxtCf");
		CNNArchitecture ca = new CNNArchitecture(); 
		
//		LeakyReLU acf = new LeakyReLU();
//		acf.setA(0.001);
		LeakyReLU acf = null;
		
		LayerConfigurator lc0 = new LayerConfigurator(LayerConfigurator.CONVOLUTION_LAYER, 
				1, true, 30, 200, 1);
		lc0.setAcf(acf);
		lc0.setFMAdaptive(true);
		ca.addLayerCfg(lc0);
		
		LayerConfigurator lc1 = new LayerConfigurator(LayerConfigurator.CONVOLUTION_LAYER, 
				120, true, 2, 2, 1);
		lc1.setAcf(acf);
		lc1.setFMAdaptive(true);
		/**different cks****/
		lc1.setCks(new int[][]{
				{2, 200},{2, 200},{2, 200},{2, 200},{2, 200},
                {2, 200},{2, 200},{2, 200},{2, 200},{2, 200},
                {2, 200},{2, 200},{2, 200},{2, 200},{2, 200},
                {2, 200},{2, 200},{2, 200},{2, 200},{2, 200},
                {2, 200},{2, 200},{2, 200},{2, 200},{2, 200},
                {2, 200},{2, 200},{2, 200},{2, 200},{2, 200},
                {2, 200},{2, 200},{2, 200},{2, 200},{2, 200},
                {2, 200},{2, 200},{2, 200},{2, 200},{2, 200},
                
                {3, 200},{3, 200},{3, 200},{3, 200},{3, 200},
                {3, 200},{3, 200},{3, 200},{3, 200},{3, 200},
                {3, 200},{3, 200},{3, 200},{3, 200},{3, 200},
                {3, 200},{3, 200},{3, 200},{3, 200},{3, 200},
                {3, 200},{3, 200},{3, 200},{3, 200},{3, 200},
                {3, 200},{3, 200},{3, 200},{3, 200},{3, 200},
                {3, 200},{3, 200},{3, 200},{3, 200},{3, 200},
                {3, 200},{3, 200},{3, 200},{3, 200},{3, 200},
                
                {4, 200},{4, 200},{4, 200},{4, 200},{4, 200},
                {4, 200},{4, 200},{4, 200},{4, 200},{4, 200},
                {4, 200},{4, 200},{4, 200},{4, 200},{4, 200},
                {4, 200},{4, 200},{4, 200},{4, 200},{4, 200},
                {4, 200},{4, 200},{4, 200},{4, 200},{4, 200},
                {4, 200},{4, 200},{4, 200},{4, 200},{4, 200},
                {4, 200},{4, 200},{4, 200},{4, 200},{4, 200},
                {4, 200},{4, 200},{4, 200},{4, 200},{4, 200},
                
		});
		ca.addLayerCfg(lc1);
		
		LayerConfigurator lc2 = new LayerConfigurator(LayerConfigurator.POOLING_LAYER, 
				120, true, 1, 1, 1);
		lc2.setAcf(acf);
		lc2.setCKAdaptive(true);
		ca.addLayerCfg(lc2);
		
		int threadsNum = 1;
		ANNCfg aNNCfg = new ANNCfg();
		aNNCfg.setDropOut(0.3);
		aNNCfg.setTesting(false);
		aNNCfg.setThreadsNum(threadsNum);
		
		LayerConfigurator lc6 = new LayerConfigurator(LayerConfigurator.ANN_LAYER, 
				120, true, 1, 1, 1);
		lc6.setAcf(acf);
		ca.addLayerCfg(lc6);
		lc6.setaNNCfg(aNNCfg);
		
		LayerConfigurator lc7 = new LayerConfigurator(LayerConfigurator.ANN_LAYER, 
				120, true, 1, 1, 1);
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
		cnnCfg.setThreadsNum(threadsNum);
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
	    TestTxtClassification testTxtClassification = new TestTxtClassification();
	    
	    String sf = "E:\\models\\cnn\\";
	    if(args != null && args.length > 0){
	        sf = args[0];
	    }

        File fsf = new File(sf);
//        if (!fsf.exists()) {            
//            sf = System.getProperty("user.dir");
//        }           
        File dir = new File(sf, "sentiment3.0");
        dir.mkdirs();
        String fname = "all-train-vector.txt";
        String tfname = "old-test-vector-1.txt";
        String tfname2 = "test-input-douban-new-vector.txt";
        String mname = "cnnCfg_1466591887401_TxtCf-20.m";
        testTxtClassification.train(dir.getAbsolutePath() + File.separator + fname, dir.getAbsolutePath() + File.separator + tfname);
        

        
        
//        testTxtClassification.test(dir.getAbsolutePath()+ File.separator + mname,
//                dir.getAbsolutePath()+ File.separator + fname, 
//                dir.getAbsolutePath()+ File.separator + tfname2);
        
        
    }
	

}
