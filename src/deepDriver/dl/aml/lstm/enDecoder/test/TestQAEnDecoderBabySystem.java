package deepDriver.dl.aml.lstm.enDecoder.test;

import deepDriver.dl.aml.lstm.enDecoder.EncoderDecoderLSTM;

public class TestQAEnDecoderBabySystem {
	
	static int qFile = 1;
	static int testA = 2;
	
	public static void main(String[] args) throws Exception {
		Encoder2DecoderSetup encoder2DecoderSetup = new Encoder2DecoderSetup();
		encoder2DecoderSetup.bootstrap(null, false);
		EncoderDecoderLSTM encoderDecoderLSTM = new EncoderDecoderLSTM(encoder2DecoderSetup.getQcfg(),
				encoder2DecoderSetup.getAcfg());
		encoderDecoderLSTM.trainModel(encoder2DecoderSetup.getQsi(), 
				encoder2DecoderSetup.getAsi(), false);
	
		
	}

}
