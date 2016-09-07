package deepDriver.dl.aml.lstm.enDecoder.test;

import deepDriver.dl.aml.lstm.attentionEnDecoder.AttentionEnDecoderLSTM;

public class TestQAEnDecoderBabySystem {
	
	static int qFile = 1;
	static int testA = 2;
	
	public static void main(String[] args) throws Exception {
		Encoder2DecoderSetup encoder2DecoderSetup = new Encoder2DecoderSetup();
		encoder2DecoderSetup.setSetupDic(true);
		encoder2DecoderSetup.bootstrap(null, false);
		AttentionEnDecoderLSTM encoderDecoderLSTM = new AttentionEnDecoderLSTM(encoder2DecoderSetup.getQcfg(),
				encoder2DecoderSetup.getAcfg());
		encoderDecoderLSTM.trainModel(encoder2DecoderSetup.getQsi(), 
				encoder2DecoderSetup.getAsi(), false);
	
		
	}

}
