package deepDriver.dl.aml.lstm.enDecoder.test;

import java.util.ArrayList;
import java.util.List;


import deepDriver.dl.aml.distribution.Fs;
import deepDriver.dl.aml.lstm.LSTMConfigurator;
import deepDriver.dl.aml.lstm.LSTMDataSet;
import deepDriver.dl.aml.lstm.LSTMWwUpdater;
import deepDriver.dl.aml.lstm.attentionEnDecoder.test.AttEn2DeSetup;
import deepDriver.dl.aml.lstm.beamSearch.BeamLayer;
import deepDriver.dl.aml.lstm.beamSearch.BeamNode;
import deepDriver.dl.aml.lstm.beamSearch.BeamSearch;
import deepDriver.dl.aml.lstm.enDecoder.EncoderDecoderLSTM;
import deepDriver.dl.aml.string.Dictionary;

public class TestEnDeQA {
	
	public static void main(String[] args) throws Exception {
		Encoder2DecoderSetup encoder2DecoderSetup = new Encoder2DecoderSetup();
		encoder2DecoderSetup.setSetupDic(false);
		encoder2DecoderSetup.bootstrap(null, false);
		String root = "D:\\6.workspace\\ANN\\lstm\\QaModel\\";
		String qfile = root+"qcfg_1472196807792_2.m";
		String afile = root+"acfg_1472196807792_2.m";
		LSTMWwUpdater checker = new LSTMWwUpdater(true, true);
		LSTMWwUpdater wWUpdater = new LSTMWwUpdater(false, true);
		LSTMConfigurator qcfg = (LSTMConfigurator) Fs.readObjFromFile(qfile);
		LSTMConfigurator acfg = (LSTMConfigurator) Fs.readObjFromFile(afile);
		System.out.println(qcfg.getLearningRate());
		System.out.println(acfg.getLearningRate());
		checker.updatewWs(qcfg, encoder2DecoderSetup.getQcfg());
		wWUpdater.updatewWs(qcfg, encoder2DecoderSetup.getQcfg());
		wWUpdater.updatewWs(acfg, encoder2DecoderSetup.getAcfg());
		System.out.println("Complete the merging..");
		checker.updatewWs(qcfg, encoder2DecoderSetup.getQcfg());

		System.out.println("q l="+encoder2DecoderSetup.getQcfg().getLearningRate()
				+ ", m="+encoder2DecoderSetup.getQcfg().getM());
		System.out.println("a l="+encoder2DecoderSetup.getAcfg().getLearningRate()
				+ ", m="+encoder2DecoderSetup.getAcfg().getM());
		
		Dictionary dic = encoder2DecoderSetup.getDic();
		String start = "到期了怎么办";////为什么登录不了//怎么我的是黄金套餐还要购买才能看电影//为什么我看视频一直缓冲//如何取消自动续费
		System.out.println(start);
		LSTMDataSet qds = dic.encodeSample(start, start.length(), qcfg.isUseThinData());
		LSTMDataSet ads = dic.encodeSample(Dictionary.EOS, 1, qcfg.isUseThinData());
		
		EncoderDecoderLSTM encoderDecoderLSTM = new EncoderDecoderLSTM(encoder2DecoderSetup.getQcfg(),
				encoder2DecoderSetup.getAcfg());
		
		List<double[][][]> ts = new ArrayList<double[][][]>();
		BeamSearch bs = encoderDecoderLSTM.beamSearch(qds, 5, ads, 41, ads.getSamples()[0][0]);
		BeamLayer bl = bs.getLayers().get(bs.getLayers().size() - 1);
		List<BeamNode> bns = bl.getBns();
		for (int j = 0; j < bns.size(); j++) {
			ts.add(new double[][][]{bs.getBnById(j, bl)});
		}
		for (int i = 0; i < ts.size(); i++) {
			String ss = dic.decoded(ts.get(i), qcfg.isUseThinData(), true);
//			as.add(ss);
			System.out.println(i+ss);
		}
	}

}
