package deepDriver.dl.aml.w2v.test;

import deepDriver.dl.aml.w2v.NativeWordStream;
import deepDriver.dl.aml.w2v.NegtiveSampling;

public class TestNegativeSamplingV2 {
	
	public static void main(String[] args) throws Exception {
		
		String f = "D:\\6.workspace\\p.NLP\\000000_0.0";
		if (args.length > 0) {
			f = args[0];
		}
		NativeWordStream nativeWordStream = new NativeWordStream(f);
		
		NegtiveSampling negtiveSampling = new NegtiveSampling();
		if (args.length > 1) {
			negtiveSampling.setThreadNum(Integer.parseInt(args[1]));
		} else {
			negtiveSampling.setThreadNum(4);
		}
		
		negtiveSampling.setLoop(100);
		long l = System.currentTimeMillis();
		negtiveSampling.w2v(nativeWordStream);
		System.out.println("time is: "+(System.currentTimeMillis() - l));
		
	}

}
