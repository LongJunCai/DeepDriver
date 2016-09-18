package deepDriver.dl.aml.contrib.MNIST;

import java.io.File;

import deepDriver.dl.aml.ann.ANNCfg;
import deepDriver.dl.aml.ann.IActivationFunction;
import deepDriver.dl.aml.cnn.ActivationFactory;
import deepDriver.dl.aml.cnn.CNNArchitecture;
import deepDriver.dl.aml.cnn.CNNConfigurator;
import deepDriver.dl.aml.cnn.ConvolutionNeuroNetwork;
import deepDriver.dl.aml.cnn.LayerConfigurator;

/**
 * Demo of Training CNN model for MNIST Hand Written Digit Recognition
 * Usage
 * Download data from http://yann.lecun.com/exdb/mnist/
 * Unzip the data and get four dataset files:
    "train-images-idx3-ubyte", "train-labels-idx1-ubyte";
    "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte";
 * Performance Measure: 97~98% Precision on test 10000 images depending on hyper parameters
 * @author xichen ding
 **/

public class MnistCNN {
	
	// kLength output target number
	public static final int TARGET_NUM = 10;
	
	public static void main(String[] args) throws Exception {
		
		MnistCNN mnistCNN = new MnistCNN();
		
		String sf = "D:\\Machine_Learning_and_Data_Mining\\DeepLearning\\MNIST";

		File fsf = new File(sf);
		if (!fsf.exists()) {			
			sf = System.getProperty("user.dir");
		}
		File dir = new File(sf, "data");
		dir.mkdirs();		
		mnistCNN.train(sf);
		// mnistCNN.trainCompare(sf);
	}
	
	public void train(String file) throws Exception {
		System.out.println("Load training file from "+file);
		MnistLoader loader = new MnistLoader(file);
		MnistDataStream mnistDS = new MnistDataStream(loader, 0);//train
		MnistDataStream tmnistDS = new MnistDataStream(loader, 1);//test
		
		CNNConfigurator cnnCfg = new CNNConfigurator(); 
		cnnCfg.setL(0.005);
		cnnCfg.setM(0.8); //Momentum update
		
		cnnCfg.setName("MNIST Classification");
		CNNArchitecture ca = new CNNArchitecture(); 
		
		IActivationFunction acf = null;
		
		//Layer 0, Input [28,28] image 28 by 28 pixel	
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
		
		//Layer 2, [2,2] pooling, output [14,14] image
		LayerConfigurator lc2 = new LayerConfigurator(LayerConfigurator.POOLING_LAYER, 
				6, true, 2, 2, 1);
		lc2.setAcf(acf);
//		lc2.setCKAdaptive(true);
		ca.addLayerCfg(lc2);
		
		//Layer 3, [5,5] feature map, output 16 [10,10] image
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
		
		//Layer 4, [2,2] featuremap, output 16  [5,5] image
		LayerConfigurator lc4 = new LayerConfigurator(LayerConfigurator.POOLING_LAYER, 
				16, true, 2, 2, 1);
		lc4.setAcf(acf);
		ca.addLayerCfg(lc4);
		
		int threadNum = 10;  // debug on just 1 thread
		
		ANNCfg aNNCfg = new ANNCfg();

		aNNCfg.setThreadsNum(threadNum);
		aNNCfg.setDropOut(0.0);
		aNNCfg.setTesting(false);
		
		IActivationFunction relu = ActivationFactory.getAf().getReLU();
		
		LayerConfigurator lc6 = new LayerConfigurator(LayerConfigurator.ANN_LAYER, 
				120, true, 1, 1, 1);
		lc6.setAcf(relu);
		ca.addLayerCfg(lc6);
		lc6.setaNNCfg(aNNCfg);

		LayerConfigurator lc8 = new LayerConfigurator(LayerConfigurator.ANN_LAYER, 
				TARGET_NUM, true, 1, 1, 1);
		lc8.setAcf(relu);
		lc8.setLast(true);
		ca.addLayerCfg(lc8);
		lc8.setaNNCfg(aNNCfg);
		
		cnnCfg.setUseBN(true);  // Batch Normalization
		cnnCfg.setUseBN(false);
		
		cnnCfg.setThreadsNum(threadNum);
		cnnCfg.setPoolingType(CNNConfigurator.MAX_POOLING_TYPE);
		ConvolutionNeuroNetwork cnn = new ConvolutionNeuroNetwork();
		cnn.construct(ca, cnnCfg);
		cnn.setDebug(true);
		
		// main Interface
		cnn.train(mnistDS, tmnistDS);
		System.out.println("Done with training.");
	}

}
