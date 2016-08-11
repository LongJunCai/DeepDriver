package deepDriver.dl.aml.random;

import java.util.Random;

public class RandomFactory {
	
	static transient Random random = new Random(System.currentTimeMillis());
	
	public static Random getRandom() {
		return random;
	}

}
