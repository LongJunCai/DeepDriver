package deepDriver.dl.aml.cnn.test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;

import deepDriver.dl.aml.ann.ANNCfg;
import deepDriver.dl.aml.cnn.CNNArchitecture;
import deepDriver.dl.aml.cnn.CNNConfigurator;
import deepDriver.dl.aml.cnn.ConvolutionNeuroNetwork;
import deepDriver.dl.aml.cnn.IDataStream;
import deepDriver.dl.aml.cnn.LayerConfigurator;
import deepDriver.dl.aml.cnn.LeakyReLU;
import deepDriver.dl.aml.cnn.img.CsvImgLoader;
import deepDriver.dl.aml.cnn.img.W2VDataStream;
import deepDriver.dl.aml.cnn.img.W2VDataStreamV2;
import deepDriver.dl.aml.cnn.img.W2VDataStreamV24Test;
import deepDriver.dl.aml.cnn.img.W2VDirectStream;
import deepDriver.dl.aml.cnn.txt.Word2vecUtil;

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
	
	ConvolutionNeuroNetwork cnn = null;
	
	public void loadM(String mfile) throws Exception {
	    if (cnn == null) {
	        System.out.println("CNN model is inited.");
	        cnn = new ConvolutionNeuroNetwork();
            cnn.readCfg(mfile);
        }	    
    }
	
	public SingleResult predictSentiment(String mfile, StringBuilder text, float [][] fws) throws Exception 
	{
	    SingleResult singleResult = new SingleResult();
	    int kLength = 3;
	    
	    if(text != null && text.length()==0)
	    {
	        singleResult.setLabel(3);
	        singleResult.setProb(1);
	        return singleResult;
	    }
	    if (fws != null && fws.length == 0) {
	        singleResult.setLabel(3);
            singleResult.setProb(1);
            return singleResult;
        }
	    IDataStream testStream = null;
	    
	    if (fws != null) {
	        testStream = new W2VDirectStream(fws, 100);
        } else {
            CsvImgLoader testLoader = new CsvImgLoader();
            testLoader.setHeader(false);
            testLoader.loadSingle(text.toString());
            testStream = new W2VDataStreamV24Test(testLoader, kLength, 200);
        }	    
                
        loadM(mfile);
        
        singleResult = cnn.predict(testStream);
        int label = singleResult.getLabel()+1;
        singleResult.setLabel(label);
        return singleResult;
	}
	
	
	
	   /**
     * 将文件中的一行文本（一条微博或豆瓣评论）转化为向量(词向量的拼接)
     * @param lineText
     * @param word2vec
     * @return StringBuilder
     */
    public StringBuilder line2vec(String lineText, Word2vecUtil word2vec )
    {   
        
        if(lineText == null)
        {
            return null;
        }
        
        String [] words = lineText.split(" ");
        DecimalFormat decimalFormat=new DecimalFormat(".00");
        StringBuilder result = new StringBuilder();
        for(String word: words)
        {
            float [] vector = word2vec.getWordVector(word);
            if(vector==null)
            {
                continue;
            }
            for(float dim : vector)
            {
                result.append(decimalFormat.format(dim * 100));
                result.append(',');
            }
        }
        return result;
    }
    
    public float [][] line2vecFloat(String lineText, Word2vecUtil word2vec )
    {   
        
        if(lineText == null)
        {
            return null;
        }
        
        String [] words = lineText.split(" ");
//        DecimalFormat decimalFormat=new DecimalFormat(".00");
        float [][] fws = new float[words.length][];
        int cnt = 0;
        for(String word: words)
        {
            float [] vector = word2vec.getWordVector(word);
            if(vector == null)
            {
                continue;
            }
            fws[cnt ++] = vector;
        }
        if (cnt != words.length) {
            float [][] nfws = new float[cnt][];
            for (int i = 0; i < nfws.length; i++) {
                nfws[i] = fws[i];
            }
            return nfws;
        }
        return fws;
    }
    
    
    
    public SingleResult getSentimentLabel(String mfile, String segInput, Word2vecUtil word2vec) throws Exception
    {
        StringBuilder linevec = null; //line2vec(segInput, word2vec);
        float [][] fws = line2vecFloat(segInput, word2vec);
        return  predictSentiment(mfile, linevec, fws);
    }
    
    
	
	public static void main(String[] args) throws Exception 
	{
//	    TestTxtClassification testTxtClassification = new TestTxtClassification();
//	    
//	    String sf = "E:\\models\\cnn\\";
//	    if(args != null && args.length > 0){
//	        sf = args[0];
//	    }
//
//        File fsf = new File(sf);
////        if (!fsf.exists()) {            
////            sf = System.getProperty("user.dir");
////        }           
//        File dir = new File(sf, "sentiment3.0");
//        dir.mkdirs();
//        
//        String fname = "vectors-mix-train-data-20160803.txt" ;
//        String tfname = "vectors-mix-test-data-20160805.txt" ;
////        String tfname2 = "test-input-douban-new-vector.txt";
////        String mname = "cnnCfg_1466591887401_TxtCf-20.m";
//        testTxtClassification.train(dir.getAbsolutePath() + File.separator + fname, dir.getAbsolutePath() + File.separator + tfname);
            
//        testTxtClassification.test(dir.getAbsolutePath()+ File.separator + mname,
//                dir.getAbsolutePath()+ File.separator + fname, 
//                dir.getAbsolutePath()+ File.separator + tfname2);
    
	    
	    
	    TestTxtClassification testTxtClassification = new TestTxtClassification();
	    SingleResult singleResult = new SingleResult();
	    
	    String mname = "data\\cnn-model1-18000-10000.m";
	    String lineText1 = "最差 的 电影 没有  之一";

	    Word2vecUtil word2vec = new Word2vecUtil();
        word2vec.loadModel("E:\\data\\word2vecResult\\vectors-20160617.bin");
        
 //       singleResult = testTxtClassification.getSentimentLabel(mname, lineText1, word2vec);
 //       System.out.println(lineText1+" label:" + singleResult.getLabel() + " probablity:"+ singleResult.getProb());
        
        
        
        
        
        FileReader reader = new FileReader("E:\\data\\新情感语料\\new-data\\shenyi-data\\data-old-3-test-seg-2016-8-2.txt");
        BufferedReader br = new BufferedReader(reader);
 //       PrintStream ps = new PrintStream("E:\\data\\新情感语料\\new-data\\shenyi-data\\output-test-2016-8-11.csv"); 
        String str = null;
        
        long startTime = System.currentTimeMillis(); 
        
        while((str = br.readLine()) != null) 
        {
            String [] words = str.split("\t");
            String input = words[2];
            singleResult = testTxtClassification.getSentimentLabel(mname, input, word2vec);
            String res = words[0] +  "\t"  + words[2]+ "\t" + words[1] + "\t"+ singleResult.getLabel() + "\t"+ singleResult.getProb();
//           System.out.println(res);
//           ps.println(res); 
        }
//        ps.close();
        br.close();
        reader.close();
        
        long endTime=System.currentTimeMillis(); //获取结束时间  
        System.out.println("程序运行时间： "+(endTime - startTime) + "ms");  
        
    }
	

}
