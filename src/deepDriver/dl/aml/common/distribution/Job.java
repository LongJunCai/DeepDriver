package deepDriver.dl.aml.common.distribution;

import java.lang.reflect.Method;

public class Job extends Thread {
	
	String mainClzz;
	
	public Job(String mainClzz) {
		super();
		this.mainClzz = mainClzz;
	}

	@Override
	public void run() {
		super.run();
		Class clzz;
		try {
			clzz = Class.forName(mainClzz);
			Object obj = clzz.newInstance();
			Method m1 = clzz.getDeclaredMethod("main", String[].class);
			m1.invoke(obj, null);
		} catch (Exception e) {
 			e.printStackTrace();
		}
		
	}

}
